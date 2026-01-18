import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
def asynccontextmanager(func):
    """@asynccontextmanager decorator.

    Typical usage:

        @asynccontextmanager
        async def some_async_generator(<arguments>):
            <setup>
            try:
                yield <value>
            finally:
                <cleanup>

    This makes this:

        async with some_async_generator(<arguments>) as <variable>:
            <body>

    equivalent to this:

        <setup>
        try:
            <variable> = <value>
            <body>
        finally:
            <cleanup>
    """

    @wraps(func)
    def helper(*args, **kwds):
        return _AsyncGeneratorContextManager(func, args, kwds)
    return helper