from __future__ import annotations
import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, Type
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, utils
@_beartype.beartype
def diagnose_call(rule: infra.Rule, *, level: infra.Level=infra.Level.NONE, diagnostic_type: Type[infra.Diagnostic]=infra.Diagnostic, format_argument: Callable[[Any], str]=formatter.format_argument, diagnostic_message_formatter: MessageFormatterType=format_message_in_text) -> Callable:

    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            common_error_message = 'diagnose_call can only be applied to callables'
            if not callable(fn):
                raise AssertionError(f'{common_error_message}. Got {type(fn)} instead of callable.')
            arg0 = args[0] if len(args) > 0 else None
            if isinstance((ctx := arg0), infra.DiagnosticContext):
                pass
            elif isinstance((ctx := getattr(arg0, 'diagnostic_context', None)), infra.DiagnosticContext):
                pass
            else:
                raise AssertionError(f"{common_error_message}. For {fn}, If it is a function, a DiagnosticContext instance must be present as the first argument. If it is a method, a DiagnosticContext instance must be present as the attribute 'diagnostic_context' of the 'self' argument.")
            diag = diagnostic_type(rule, level, diagnostic_message_formatter(fn, *args, **kwargs))
            stack: Optional[infra.Stack] = None
            if len(diag.stacks) > 0:
                stack = diag.stacks[0]
                stack.frames.pop(0)
            fn_location = utils.function_location(fn)
            diag.locations.insert(0, fn_location)
            if stack is not None:
                stack.frames.insert(0, infra.StackFrame(location=fn_location))
            with diag.log_section(logging.INFO, 'Function Signature'):
                diag.log(logging.INFO, '%s', formatter.LazyString(format_function_signature_in_markdown, fn, args, kwargs, format_argument))
            return_values: Any = None
            with ctx.add_inflight_diagnostic(diag) as diag:
                try:
                    return_values = fn(*args, **kwargs)
                    with diag.log_section(logging.INFO, 'Return values'):
                        diag.log(logging.INFO, '%s', formatter.LazyString(format_return_values_in_markdown, return_values, format_argument))
                    return return_values
                except Exception as e:
                    diag.log_source_exception(logging.ERROR, e)
                    diag.level = infra.Level.ERROR
                finally:
                    ctx.log_and_raise_if_error(diag)
        return wrapper
    return decorator