import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar
def completed_future(data: T) -> AwaitableFuture[T]:
    """Return a future with the given data as its result."""
    f = AwaitableFuture[T]()
    f.set_result(data)
    return f