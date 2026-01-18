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
def get_function_first_arg_dict_keys(func: Callable) -> Optional[List[str]]:
    """Get the keys of the first argument of a function if it is a dict."""
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = IsFunctionArgDict()
        visitor.visit(tree)
        return list(visitor.keys) if visitor.keys else None
    except (SyntaxError, TypeError, OSError):
        return None