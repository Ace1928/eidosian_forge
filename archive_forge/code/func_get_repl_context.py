import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
def get_repl_context() -> Any:
    """Get the notebook REPL context if running inside a Databricks notebook.
    Returns None otherwise.
    """
    try:
        from dbruntime.databricks_repl_context import get_context
        return get_context()
    except ImportError:
        raise ImportError('Cannot access dbruntime, not running inside a Databricks notebook.')