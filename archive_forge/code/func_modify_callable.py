from typing import Callable, List, Optional
from mypy.plugin import FunctionContext, Plugin
from mypy.types import CallableType, get_proper_type, Instance, Overloaded, Type
def modify_callable(func_type: CallableType, ctx: FunctionContext) -> Type:
    ret_type = get_proper_type(func_type.ret_type)
    if not (isinstance(ret_type, Instance) and ret_type.type.name == 'Coroutine'):
        if not func_type.implicit:
            ctx.api.msg.fail(f'expected return type Awaitable[T], got {ret_type}', ctx.context)
        return ctx.default_return_type
    result_type = ret_type.args[-1]
    return func_type.copy_modified(ret_type=result_type)