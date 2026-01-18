from contextlib import ExitStack, contextmanager
from typing import ContextManager, Generator, TypeVar
@contextmanager
def main_context(self) -> Generator[None, None, None]:
    assert not self._in_main_context
    self._in_main_context = True
    try:
        with self._main_context:
            yield
    finally:
        self._in_main_context = False