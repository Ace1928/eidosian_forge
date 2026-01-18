from __future__ import annotations
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.friendli import BaseFriendli
def get_role(message: BaseMessage) -> str:
    """Get role of the message.

    Args:
        message (BaseMessage): The message object.

    Raises:
        ValueError: Raised when the message is of an unknown type.

    Returns:
        str: The role of the message.
    """
    if isinstance(message, ChatMessage) or isinstance(message, HumanMessage):
        return 'user'
    if isinstance(message, AIMessage):
        return 'assistant'
    if isinstance(message, SystemMessage):
        return 'system'
    raise ValueError(f'Got unknown type {message}')