from __future__ import annotations
import os
import re
import inspect
import functools
from typing import (
from pathlib import Path
from typing_extensions import TypeGuard
import sniffio
from .._types import Headers, NotGiven, FileTypes, NotGivenOr, HeadersLike
from .._compat import parse_date as parse_date, parse_datetime as parse_datetime
def required_args(*variants: Sequence[str]) -> Callable[[CallableT], CallableT]:
    """Decorator to enforce a given set of arguments or variants of arguments are passed to the decorated function.

    Useful for enforcing runtime validation of overloaded functions.

    Example usage:
    ```py
    @overload
    def foo(*, a: str) -> str:
        ...


    @overload
    def foo(*, b: bool) -> str:
        ...


    # This enforces the same constraints that a static type checker would
    # i.e. that either a or b must be passed to the function
    @required_args(["a"], ["b"])
    def foo(*, a: str | None = None, b: bool | None = None) -> str:
        ...
    ```
    """

    def inner(func: CallableT) -> CallableT:
        params = inspect.signature(func).parameters
        positional = [name for name, param in params.items() if param.kind in {param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD}]

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            given_params: set[str] = set()
            for i, _ in enumerate(args):
                try:
                    given_params.add(positional[i])
                except IndexError:
                    raise TypeError(f'{func.__name__}() takes {len(positional)} argument(s) but {len(args)} were given') from None
            for key in kwargs.keys():
                given_params.add(key)
            for variant in variants:
                matches = all((param in given_params for param in variant))
                if matches:
                    break
            else:
                if len(variants) > 1:
                    variations = human_join(['(' + human_join([quote(arg) for arg in variant], final='and') + ')' for variant in variants])
                    msg = f'Missing required arguments; Expected either {variations} arguments to be given'
                else:
                    missing = list(set(variants[0]) - given_params)
                    if len(missing) > 1:
                        msg = f'Missing required arguments: {human_join([quote(arg) for arg in missing])}'
                    else:
                        msg = f'Missing required argument: {quote(missing[0])}'
                raise TypeError(msg)
            return func(*args, **kwargs)
        return wrapper
    return inner