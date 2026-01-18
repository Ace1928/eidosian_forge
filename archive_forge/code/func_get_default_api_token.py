import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
def get_default_api_token() -> str:
    """Get the default Databricks personal access token.
    Raises an error if the token cannot be automatically determined.
    """
    if (api_token := os.getenv('DATABRICKS_TOKEN')):
        return api_token
    try:
        api_token = get_repl_context().apiToken
        if not api_token:
            raise ValueError("context doesn't contain apiToken.")
    except Exception as e:
        raise ValueError(f"api_token was not set and cannot be automatically inferred. Set environment variable 'DATABRICKS_TOKEN'. Received error: {e}")
    return api_token