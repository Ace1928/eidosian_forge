from __future__ import annotations
import asyncio
import copy
import threading
from collections import defaultdict
from typing import (
from uuid import UUID
import jsonpatch  # type: ignore[import]
from typing_extensions import NotRequired, TypedDict
from langchain_core.load import dumps
from langchain_core.load.load import load
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import Input, Output
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.memory_stream import _MemoryStream
from langchain_core.tracers.schemas import Run
def include_run(self, run: Run) -> bool:
    if run.id == self.root_id:
        return False
    run_tags = run.tags or []
    if self.include_names is None and self.include_types is None and (self.include_tags is None):
        include = True
    else:
        include = False
    if self.include_names is not None:
        include = include or run.name in self.include_names
    if self.include_types is not None:
        include = include or run.run_type in self.include_types
    if self.include_tags is not None:
        include = include or any((tag in self.include_tags for tag in run_tags))
    if self.exclude_names is not None:
        include = include and run.name not in self.exclude_names
    if self.exclude_types is not None:
        include = include and run.run_type not in self.exclude_types
    if self.exclude_tags is not None:
        include = include and all((tag not in self.exclude_tags for tag in run_tags))
    return include