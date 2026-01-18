from __future__ import annotations
import asyncio
import functools
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.openai_tools import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
from langchain_community.llms.tongyi import (
def convert_message_chunk_to_message(message_chunk: BaseMessageChunk) -> BaseMessage:
    """Convert a message chunk to a message."""
    if isinstance(message_chunk, HumanMessageChunk):
        return HumanMessage(content=message_chunk.content)
    elif isinstance(message_chunk, AIMessageChunk):
        return AIMessage(content=message_chunk.content)
    elif isinstance(message_chunk, SystemMessageChunk):
        return SystemMessage(content=message_chunk.content)
    elif isinstance(message_chunk, ChatMessageChunk):
        return ChatMessage(role=message_chunk.role, content=message_chunk.content)
    else:
        raise TypeError(f'Got unknown type {message_chunk}')