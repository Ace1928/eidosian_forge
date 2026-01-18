import functools
import inspect
import warnings
import sys
from typing import Any, Callable, TypeVar, cast
@functools.wraps(func)
def generator_context(*args, **kwargs):
    gen = func(*args, **kwargs)
    try:
        with ctx_factory():
            response = gen.send(None)
        while True:
            try:
                request = (yield response)
            except GeneratorExit:
                with ctx_factory():
                    gen.close()
                raise
            except BaseException:
                with ctx_factory():
                    response = gen.throw(*sys.exc_info())
            else:
                with ctx_factory():
                    response = gen.send(request)
    except StopIteration as e:
        return e.value