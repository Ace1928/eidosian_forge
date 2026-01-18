import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import (
@property
def llm(self) -> bool:
    return self.task in ('llm/v1/chat', 'llm/v1/completions', 'llama2/chat')