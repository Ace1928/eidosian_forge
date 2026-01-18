import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
def call_JitFunction(self, fn: JITFunction, args, kwargs):
    args = inspect.getcallargs(fn.fn, *args, **kwargs)
    args = [args[name] for name in fn.arg_names]
    args = [arg if _is_triton_tensor(arg) else constexpr(arg) for arg in args]
    attributes = dict()
    constexprs = [i for i, arg in enumerate(args) if _is_constexpr(arg)]
    constants = {i: args[i] for i in constexprs}
    args = [None if i in constexprs else arg for i, arg in enumerate(args)]
    arg_vals = [arg.handle for arg in args if arg is not None]
    arg_types = [arg.type for arg in args if arg is not None]
    fn_name = mangle_fn(fn.__name__, arg_types, constants)
    if not self.module.has_function(fn_name):
        prototype = language.function_type([], arg_types)
        gscope = sys.modules[fn.fn.__module__].__dict__
        debug = self.debug if fn.debug is None else fn.debug
        file_name, begin_line = _get_fn_file_line(fn)
        generator = CodeGenerator(self.context, prototype, gscope, attributes, constants, module=self.module, function_name=fn_name, function_types=self.function_ret_types, debug=debug, noinline=fn.noinline, file_name=file_name, begin_line=begin_line, target=self.builder.target)
        generator.visit(fn.parse())
        callee_ret_type = generator.last_ret_type
        self.function_ret_types[fn_name] = callee_ret_type
    else:
        callee_ret_type = self.function_ret_types[fn_name]
    symbol = self.module.get_function(fn_name)
    call_op = self.builder.call(symbol, arg_vals)
    if call_op.get_num_results() == 0 or callee_ret_type is None:
        return None
    elif call_op.get_num_results() == 1:
        return tensor(call_op.get_result(0), callee_ret_type)
    else:
        results = []
        for i in range(call_op.get_num_results()):
            results.append(tensor(call_op.get_result(i), callee_ret_type[i]))
        return tuple(results)