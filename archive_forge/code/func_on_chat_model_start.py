import os
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
from langchain_core.outputs import LLMResult
def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> None:
    self.messages = [_convert_message_to_dict(message) for message in messages[0]]
    self.prompt = self.messages[-1]['content']