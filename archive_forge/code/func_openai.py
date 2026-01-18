import functools
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from outlines.base import vectorize
from outlines.caching import cache
def openai(model_name: str, api_key: Optional[str]=None, config: Optional[OpenAIConfig]=None):
    try:
        import tiktoken
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("The `openai` and `tiktoken` libraries needs to be installed in order to use Outlines' OpenAI integration.")
    if config is not None:
        config = replace(config, model=model_name)
    else:
        config = OpenAIConfig(model=model_name)
    client = AsyncOpenAI(api_key=api_key)
    tokenizer = tiktoken.encoding_for_model(model_name)
    return OpenAI(client, config, tokenizer)