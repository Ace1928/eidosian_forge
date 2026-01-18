from __future__ import annotations
import asyncio
import collections
import inspect
import threading
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, wait
from contextvars import copy_context
from functools import wraps
from itertools import groupby, tee
from operator import itemgetter
from typing import (
from typing_extensions import Literal, get_args
from langchain_core._api import beta_decorator
from langchain_core.load.dump import dumpd
from langchain_core.load.serializable import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.schema import EventData, StreamEvent
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
def batch_as_completed(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: bool=False, **kwargs: Optional[Any]) -> Iterator[Tuple[int, Union[Output, Exception]]]:
    if isinstance(config, list):
        configs = cast(List[RunnableConfig], [self._merge_configs(conf) for conf in config])
    else:
        configs = [self._merge_configs(config) for _ in range(len(inputs))]
    if return_exceptions:
        yield from self.bound.batch_as_completed(inputs, configs, return_exceptions=return_exceptions, **{**self.kwargs, **kwargs})
    else:
        yield from self.bound.batch_as_completed(inputs, configs, return_exceptions=return_exceptions, **{**self.kwargs, **kwargs})