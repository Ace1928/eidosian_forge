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
def get_lambda_source(func: Callable) -> Optional[str]:
    """Get the source code of a lambda function.

    Args:
        func: a callable that can be a lambda function

    Returns:
        str: the source code of the lambda function
    """
    try:
        name = func.__name__ if func.__name__ != '<lambda>' else None
    except AttributeError:
        name = None
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = GetLambdaSource()
        visitor.visit(tree)
        return visitor.source if visitor.count == 1 else name
    except (SyntaxError, TypeError, OSError):
        return name