import os
import sys
from typing import (  # noqa: F401
def assert_never(inp: NoReturn, raise_error: bool=True, exc: Union[Exception, None]=None) -> None:
    """For use in exhaustive checking of literal or Enum in if/else chain.

    Should only be reached if all members not handled OR attempt to pass non-members through chain.

    If all members handled, type is Empty. Otherwise, will cause mypy error.

    If non-members given, should cause mypy error at variable creation.

    If raise_error is True, will also raise AssertionError or the Exception passed to exc.
    """
    if raise_error:
        if exc is None:
            raise ValueError(f'An unhandled Literal ({inp}) in an if/else chain was found')
        else:
            raise exc