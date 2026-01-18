from __future__ import annotations
import logging
import os
import sys
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import Runnable
from langchain_core.utils import (
from langchain_community.adapters.openai import (
from langchain_community.utils.openai import is_openai_v1
def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
    """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
    if sys.version_info[1] <= 7:
        return super().get_num_tokens_from_messages(messages)
    model, encoding = self._get_encoding_model()
    if model.startswith('gpt-3.5-turbo-0301'):
        tokens_per_message = 4
        tokens_per_name = -1
    elif model.startswith('gpt-3.5-turbo') or model.startswith('gpt-4'):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f'get_num_tokens_from_messages() is not presently implemented for model {model}.See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.')
    num_tokens = 0
    messages_dict = [convert_message_to_dict(m) for m in messages]
    for message in messages_dict:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == 'name':
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens