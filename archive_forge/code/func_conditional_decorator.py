import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str
from langchain_core.utils.env import get_from_dict_or_env
def conditional_decorator(condition: bool, decorator: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Conditionally apply a decorator.

    Args:
        condition: A boolean indicating whether to apply the decorator.
        decorator: A decorator function.

    Returns:
        A decorator function.
    """

    def actual_decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if condition:
            return decorator(func)
        return func
    return actual_decorator