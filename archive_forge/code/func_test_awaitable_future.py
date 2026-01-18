from concurrent.futures import Future
from typing import Any, Callable, Optional
import pytest
import duet
def test_awaitable_future():
    assert isinstance(duet.awaitable(Future()), duet.AwaitableFuture)