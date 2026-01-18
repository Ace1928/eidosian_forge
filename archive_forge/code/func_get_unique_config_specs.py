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
def get_unique_config_specs(specs: Iterable[ConfigurableFieldSpec]) -> List[ConfigurableFieldSpec]:
    """Get the unique config specs from a sequence of config specs."""
    grouped = groupby(sorted(specs, key=lambda s: (s.id, *(s.dependencies or []))), lambda s: s.id)
    unique: List[ConfigurableFieldSpec] = []
    for id, dupes in grouped:
        first = next(dupes)
        others = list(dupes)
        if len(others) == 0:
            unique.append(first)
        elif all((o == first for o in others)):
            unique.append(first)
        else:
            raise ValueError(f'RunnableSequence contains conflicting config specsfor {id}: {[first] + others}')
    return unique