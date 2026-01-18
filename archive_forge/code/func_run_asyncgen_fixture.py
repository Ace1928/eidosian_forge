from __future__ import annotations
import types
from abc import ABCMeta, abstractmethod
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable
from typing import Any, TypeVar
@abstractmethod
def run_asyncgen_fixture(self, fixture_func: Callable[..., AsyncGenerator[_T, Any]], kwargs: dict[str, Any]) -> Iterable[_T]:
    """
        Run an async generator fixture.

        :param fixture_func: the fixture function
        :param kwargs: keyword arguments to call the fixture function with
        :return: an iterator yielding the value yielded from the async generator
        """