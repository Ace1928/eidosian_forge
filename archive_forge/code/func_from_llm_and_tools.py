from typing import Any, List, Optional, Sequence, Tuple
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.chat.output_parser import ChatOutputParser
from langchain.agents.chat.prompt import (
from langchain.agents.utils import validate_tools_single_input
from langchain.chains.llm import LLMChain
@classmethod
def from_llm_and_tools(cls, llm: BaseLanguageModel, tools: Sequence[BaseTool], callback_manager: Optional[BaseCallbackManager]=None, output_parser: Optional[AgentOutputParser]=None, system_message_prefix: str=SYSTEM_MESSAGE_PREFIX, system_message_suffix: str=SYSTEM_MESSAGE_SUFFIX, human_message: str=HUMAN_MESSAGE, format_instructions: str=FORMAT_INSTRUCTIONS, input_variables: Optional[List[str]]=None, **kwargs: Any) -> Agent:
    """Construct an agent from an LLM and tools."""
    cls._validate_tools(tools)
    prompt = cls.create_prompt(tools, system_message_prefix=system_message_prefix, system_message_suffix=system_message_suffix, human_message=human_message, format_instructions=format_instructions, input_variables=input_variables)
    llm_chain = LLMChain(llm=llm, prompt=prompt, callback_manager=callback_manager)
    tool_names = [tool.name for tool in tools]
    _output_parser = output_parser or cls._get_default_output_parser()
    return cls(llm_chain=llm_chain, allowed_tools=tool_names, output_parser=_output_parser, **kwargs)