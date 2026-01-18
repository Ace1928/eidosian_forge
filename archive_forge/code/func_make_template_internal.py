import functools
import traceback
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def make_template_internal(name_, func_, create_scope_now_=False, unique_name_=None, custom_getter_=None, create_graph_function_=False, **kwargs):
    """Make a template, optionally compiling func_ into a graph function.

  See `make_template` for full documentation.

  Args:
    name_: A name for the scope created by this template. If necessary, the name
      will be made unique by appending `_N` to the name.
    func_: The function to wrap.
    create_scope_now_: Boolean controlling whether the scope should be created
      when the template is constructed or when the template is called. Default
      is False, meaning the scope is created when the template is called.
    unique_name_: When used, it overrides name_ and is not made unique. If a
      template of the same scope/unique_name already exists and reuse is false,
      an error is raised. Defaults to None. If executing eagerly, must be None.
    custom_getter_: Optional custom getter for variables used in `func_`. See
      the `tf.compat.v1.get_variable` `custom_getter` documentation for more
      information.
    create_graph_function_: When True, `func_` will be executed as a graph
      function. This implies that `func_` must satisfy the properties that
      `function.defun` requires of functions: See the documentation of
        `function.defun` for details. When executing eagerly, setting this flag
        to True can improve performance. Regardless of whether eager execution
        is enabled, enabling this flag gives the caller access to graph-function
        semantics, i.e., accesses to variables are totally ordered and
        side-effecting ops are not pruned.
    **kwargs: Keyword arguments to apply to `func_`.

  Returns:
    A function to encapsulate a set of variables which should be created once
    and reused. An enclosing scope will be created either when `make_template`
    is called or when the result is called, depending on the value of
    `create_scope_now_`. Regardless of the value, the first time the template
    is called it will enter the scope with no reuse, and call `func_` to create
    variables, which are guaranteed to be unique. All subsequent calls will
    re-enter the scope and reuse those variables.

  Raises:
    ValueError: if `name_` is None.
    ValueError: if `unique_name_` is not None and eager execution is enabled.
  """
    if kwargs:
        func_ = functools.partial(func_, **kwargs)
    if context.executing_eagerly():
        if unique_name_ is not None:
            raise ValueError('unique_name_ cannot be used when eager execution is enabled.')
        return EagerTemplate(name_, func_, create_scope_now=create_scope_now_, custom_getter=custom_getter_, create_graph_function=create_graph_function_)
    return Template(name_, func_, create_scope_now=create_scope_now_, unique_name=unique_name_, custom_getter=custom_getter_, create_graph_function=create_graph_function_)