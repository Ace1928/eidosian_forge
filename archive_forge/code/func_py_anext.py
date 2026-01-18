import asyncio
import inspect
from collections import deque
from typing import (
def py_anext(iterator: AsyncIterator[T], default: Union[T, Any]=_no_default) -> Awaitable[Union[T, None, Any]]:
    """Pure-Python implementation of anext() for testing purposes.

    Closely matches the builtin anext() C implementation.
    Can be used to compare the built-in implementation of the inner
    coroutines machinery to C-implementation of __anext__() and send()
    or throw() on the returned generator.
    """
    try:
        __anext__ = cast(Callable[[AsyncIterator[T]], Awaitable[T]], type(iterator).__anext__)
    except AttributeError:
        raise TypeError(f'{iterator!r} is not an async iterator')
    if default is _no_default:
        return __anext__(iterator)

    async def anext_impl() -> Union[T, Any]:
        try:
            return await __anext__(iterator)
        except StopAsyncIteration:
            return default
    return anext_impl()