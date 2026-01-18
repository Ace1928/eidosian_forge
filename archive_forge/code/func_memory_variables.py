from typing import Any, Dict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain.agents.format_scratchpad.openai_functions import (
from langchain.memory.chat_memory import BaseChatMemory
@property
def memory_variables(self) -> List[str]:
    """Will always return list of memory variables.

        :meta private:
        """
    return [self.memory_key]