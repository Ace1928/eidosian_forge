from __future__ import annotations
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.friendli import BaseFriendli
def get_chat_request(messages: List[BaseMessage]) -> Dict[str, Any]:
    """Get a request of the Friendli chat API.

    Args:
        messages (List[BaseMessage]): Messages comprising the conversation so far.

    Returns:
        Dict[str, Any]: The request for the Friendli chat API.
    """
    return {'messages': [{'role': get_role(message), 'content': message.content} for message in messages]}