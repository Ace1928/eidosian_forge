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
def create_prompt(cls, tools: Sequence[BaseTool], system_message_prefix: str=SYSTEM_MESSAGE_PREFIX, system_message_suffix: str=SYSTEM_MESSAGE_SUFFIX, human_message: str=HUMAN_MESSAGE, format_instructions: str=FORMAT_INSTRUCTIONS, input_variables: Optional[List[str]]=None) -> BasePromptTemplate:
    tool_strings = '\n'.join([f'{tool.name}: {tool.description}' for tool in tools])
    tool_names = ', '.join([tool.name for tool in tools])
    format_instructions = format_instructions.format(tool_names=tool_names)
    template = '\n\n'.join([system_message_prefix, tool_strings, format_instructions, system_message_suffix])
    messages = [SystemMessagePromptTemplate.from_template(template), HumanMessagePromptTemplate.from_template(human_message)]
    if input_variables is None:
        input_variables = ['input', 'agent_scratchpad']
    return ChatPromptTemplate(input_variables=input_variables, messages=messages)