from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar
def run_cleanup(self) -> None:
    if self.cleanup is not None:
        self.cleanup(self.key, self.value)
    self.cleanup = None
    del self.value