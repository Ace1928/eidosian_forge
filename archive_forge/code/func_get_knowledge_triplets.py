from typing import Any, Dict, List, Type, Union
from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.graphs.networkx_graph import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import (
from langchain.memory.utils import get_prompt_input_key
def get_knowledge_triplets(self, input_string: str) -> List[KnowledgeTriple]:
    chain = LLMChain(llm=self.llm, prompt=self.knowledge_extraction_prompt)
    buffer_string = get_buffer_string(self.chat_memory.messages[-self.k * 2:], human_prefix=self.human_prefix, ai_prefix=self.ai_prefix)
    output = chain.predict(history=buffer_string, input=input_string, verbose=True)
    knowledge = parse_triples(output)
    return knowledge