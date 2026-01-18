from __future__ import annotations
from typing import Any, Dict, List, Type
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import SUMMARY_PROMPT
@classmethod
def from_messages(cls, llm: BaseLanguageModel, chat_memory: BaseChatMessageHistory, *, summarize_step: int=2, **kwargs: Any) -> ConversationSummaryMemory:
    obj = cls(llm=llm, chat_memory=chat_memory, **kwargs)
    for i in range(0, len(obj.chat_memory.messages), summarize_step):
        obj.buffer = obj.predict_new_summary(obj.chat_memory.messages[i:i + summarize_step], obj.buffer)
    return obj