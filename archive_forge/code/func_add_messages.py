from __future__ import annotations
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def add_messages(self, messages: Sequence[BaseMessage]) -> None:
    """Append the messages to the Zep memory history"""
    from zep_python import Memory, Message
    zep_messages = [Message(content=message.content, role=message.type, metadata=message.additional_kwargs.get('metadata', None)) for message in messages]
    zep_memory = Memory(messages=zep_messages)
    self.zep_client.memory.add_memory(self.session_id, zep_memory)