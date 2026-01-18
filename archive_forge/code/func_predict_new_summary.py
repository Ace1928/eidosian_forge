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
def predict_new_summary(self, messages: List[BaseMessage], existing_summary: str) -> str:
    new_lines = get_buffer_string(messages, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix)
    chain = LLMChain(llm=self.llm, prompt=self.prompt)
    return chain.predict(summary=existing_summary, new_lines=new_lines)