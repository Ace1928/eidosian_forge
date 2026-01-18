import logging
from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional
from langchain_community.utilities.redis import get_client
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import (
from langchain.memory.utils import get_prompt_input_key
@property
def full_key_prefix(self) -> str:
    return f'{self.key_prefix}:{self.session_id}'