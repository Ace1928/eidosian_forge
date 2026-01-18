import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Extra
from langchain_community.chat_models.anthropic import (
from langchain_community.chat_models.meta import convert_messages_to_prompt_llama
from langchain_community.llms.bedrock import BedrockBase
from langchain_community.utilities.anthropic import (
@classmethod
def convert_messages_to_prompt(cls, provider: str, messages: List[BaseMessage]) -> str:
    if provider == 'anthropic':
        prompt = convert_messages_to_prompt_anthropic(messages=messages)
    elif provider == 'meta':
        prompt = convert_messages_to_prompt_llama(messages=messages)
    elif provider == 'mistral':
        prompt = convert_messages_to_prompt_mistral(messages=messages)
    elif provider == 'amazon':
        prompt = convert_messages_to_prompt_anthropic(messages=messages, human_prompt='\n\nUser:', ai_prompt='\n\nBot:')
    else:
        raise NotImplementedError(f'Provider {provider} model does not support chat.')
    return prompt