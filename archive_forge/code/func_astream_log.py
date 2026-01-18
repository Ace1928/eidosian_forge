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
@overload
def astream_log(self, input: Any, config: Optional[RunnableConfig]=None, *, diff: Literal[False], with_streamed_output_list: bool=True, include_names: Optional[Sequence[str]]=None, include_types: Optional[Sequence[str]]=None, include_tags: Optional[Sequence[str]]=None, exclude_names: Optional[Sequence[str]]=None, exclude_types: Optional[Sequence[str]]=None, exclude_tags: Optional[Sequence[str]]=None, **kwargs: Any) -> AsyncIterator[RunLog]:
    ...