import abc
import collections
import contextlib
import functools
import inspect
from concurrent.futures import CancelledError
from typing import (
import duet.impl as impl
from duet.aitertools import aenumerate, aiter, AnyIterable, AsyncCollector
from duet.futuretools import AwaitableFuture
@property
@abc.abstractmethod
def limiter(self) -> Limiter:
    pass