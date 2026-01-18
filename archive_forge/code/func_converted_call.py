import functools
import importlib
import inspect
import os
import sys
import textwrap
import traceback
from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.converters import asserts
from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.converters import conditional_expressions
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.converters import directives
from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.converters import logical_expressions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.converters import slices
from tensorflow.python.autograph.converters import variables
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.autograph.core import unsupported_features_checker
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import error_utils
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def converted_call(f, args, kwargs, caller_fn_scope=None, options=None):
    """Converts a function call inline.

  For internal use only.

  Note: The argument list is optimized for readability of generated code, which
  may look like this:

    ag__.converted_call(f, (arg1, arg2), None, fscope)
    ag__.converted_call(f, (), dict(arg1=val1, **kwargs), fscope)
    ag__.converted_call(f, (arg1, arg2) + varargs, dict(**kwargs), lscope)

  Args:
    f: The function to convert.
    args: Tuple, the original positional arguments of f
    kwargs: Optional[Dict], the original keyword arguments of f
    caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
      scope of the converted function in which this call was originally made.
    options: Optional[converter.ConversionOptions], conversion options. If not
      specified, the value of caller_fn_scope.callopts is used. Either options
      or caller_fn_scope must be present.

  Returns:
    Any, the result of executing a possibly-converted `f` with the given
      arguments.
  """
    logging.log(1, 'Converted call: %s\n    args: %s\n    kwargs: %s\n', f, args, kwargs)
    if options is None:
        if caller_fn_scope is None:
            raise ValueError('either caller_fn_scope or options must have a value')
        options = caller_fn_scope.callopts
    if conversion.is_in_allowlist_cache(f, options):
        logging.log(2, 'Allowlisted %s: from cache', f)
        return _call_unconverted(f, args, kwargs, options, False)
    if ag_ctx.control_status_ctx().status == ag_ctx.Status.DISABLED:
        logging.log(2, 'Allowlisted: %s: AutoGraph is disabled in context', f)
        return _call_unconverted(f, args, kwargs, options, False)
    if is_autograph_artifact(f):
        logging.log(2, 'Permanently allowed: %s: AutoGraph artifact', f)
        return _call_unconverted(f, args, kwargs, options)
    if isinstance(f, functools.partial):
        new_kwargs = {}
        if f.keywords is not None:
            new_kwargs = f.keywords.copy()
        if kwargs is not None:
            new_kwargs.update(kwargs)
        new_args = f.args + args
        logging.log(3, 'Forwarding call of partial %s with\n%s\n%s\n', f, new_args, new_kwargs)
        return converted_call(f.func, new_args, new_kwargs, caller_fn_scope=caller_fn_scope, options=options)
    if inspect_utils.isbuiltin(f):
        if f is eval:
            return py_builtins.eval_in_original_context(f, args, caller_fn_scope)
        if f is super:
            return py_builtins.super_in_original_context(f, args, caller_fn_scope)
        if f is globals:
            return py_builtins.globals_in_original_context(caller_fn_scope)
        if f is locals:
            return py_builtins.locals_in_original_context(caller_fn_scope)
        if kwargs:
            return py_builtins.overload_of(f)(*args, **kwargs)
        else:
            return py_builtins.overload_of(f)(*args)
    if conversion.is_unsupported(f):
        return _call_unconverted(f, args, kwargs, options)
    if not options.user_requested and conversion.is_allowlisted(f):
        return _call_unconverted(f, args, kwargs, options)
    if not options.internal_convert_user_code:
        return _call_unconverted(f, args, kwargs, options)
    try:
        if inspect.ismethod(f) or inspect.isfunction(f):
            target_entity = f
            effective_args = args
            f_self = getattr(f, '__self__', None)
            if f_self is not None:
                if isinstance(f_self, tf_method_target.TfMethodTarget):
                    f_self = f_self.target
                effective_args = (f_self,) + effective_args
        elif hasattr(f, '__class__') and hasattr(f.__class__, '__call__'):
            target_entity = f.__class__.__call__
            effective_args = (f,) + args
        else:
            target_entity = f
            raise NotImplementedError('unknown callable type "%s"' % type(f))
    except Exception as e:
        logging.log(1, 'Error transforming entity %s', target_entity, exc_info=True)
        if is_autograph_strict_conversion_mode():
            raise
        return _fall_back_unconverted(f, args, kwargs, options, e)
    if not hasattr(target_entity, '__code__'):
        logging.log(2, 'Permanently allowed: %s: native binding', target_entity)
        return _call_unconverted(f, args, kwargs, options)
    elif hasattr(target_entity.__code__, 'co_filename') and target_entity.__code__.co_filename == '<string>':
        logging.log(2, 'Permanently allowed: %s: dynamic code (exec?)', target_entity)
        return _call_unconverted(f, args, kwargs, options)
    try:
        program_ctx = converter.ProgramContext(options=options)
        converted_f = _convert_actual(target_entity, program_ctx)
        if logging.has_verbosity(2):
            _log_callargs(converted_f, effective_args, kwargs)
    except Exception as e:
        logging.log(1, 'Error transforming entity %s', target_entity, exc_info=True)
        if is_autograph_strict_conversion_mode():
            raise
        return _fall_back_unconverted(f, args, kwargs, options, e)
    with StackTraceMapper(converted_f), tf_stack.CurrentModuleFilter():
        try:
            if kwargs is not None:
                result = converted_f(*effective_args, **kwargs)
            else:
                result = converted_f(*effective_args)
        except Exception as e:
            _attach_error_metadata(e, converted_f)
            raise
    return result