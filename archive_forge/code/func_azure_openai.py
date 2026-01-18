import functools
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from outlines.base import vectorize
from outlines.caching import cache
def azure_openai(deployment_name: str, azure_endpoint: Optional[str]=None, api_version: Optional[str]=None, api_key: Optional[str]=None, config: Optional[OpenAIConfig]=None):
    try:
        import tiktoken
        from openai import AzureAsyncOpenAI
    except ImportError:
        raise ImportError("The `openai` and `tiktoken` libraries needs to be installed in order to use Outlines' Azure OpenAI integration.")
    if config is not None:
        config = replace(config, model=deployment_name)
    if config is None:
        config = OpenAIConfig(model=deployment_name)
    client = AzureAsyncOpenAI(azure_endpoint=azure_endpoint, api_version=api_version, api_key=api_key)
    tokenizer = tiktoken.encoding_for_model(deployment_name)
    return OpenAI(client, config, tokenizer)