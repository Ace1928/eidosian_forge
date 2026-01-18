from __future__ import annotations
import ast
import asyncio
import inspect
import textwrap
from functools import lru_cache
from inspect import signature
from itertools import groupby
from typing import (
from langchain_core.pydantic_v1 import BaseConfig, BaseModel
from langchain_core.pydantic_v1 import create_model as _create_model_base
from langchain_core.runnables.schema import StreamEvent
def include_event(self, event: StreamEvent, root_type: str) -> bool:
    """Determine whether to include an event."""
    if self.include_names is None and self.include_types is None and (self.include_tags is None):
        include = True
    else:
        include = False
    event_tags = event.get('tags') or []
    if self.include_names is not None:
        include = include or event['name'] in self.include_names
    if self.include_types is not None:
        include = include or root_type in self.include_types
    if self.include_tags is not None:
        include = include or any((tag in self.include_tags for tag in event_tags))
    if self.exclude_names is not None:
        include = include and event['name'] not in self.exclude_names
    if self.exclude_types is not None:
        include = include and root_type not in self.exclude_types
    if self.exclude_tags is not None:
        include = include and all((tag not in self.exclude_tags for tag in event_tags))
    return include