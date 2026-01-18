from typing import Any, Dict, List, Union
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain.memory.chat_memory import BaseChatMemory
@property
def buffer_as_messages(self) -> List[BaseMessage]:
    """Exposes the buffer as a list of messages in case return_messages is True."""
    return self.chat_memory.messages[-self.k * 2:] if self.k > 0 else []