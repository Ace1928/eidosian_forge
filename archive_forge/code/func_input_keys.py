from typing import Any, List, Sequence, Tuple, Union
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain.agents.agent import BaseSingleActionAgent
from langchain.agents.format_scratchpad import format_xml
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain.agents.xml.prompt import agent_instructions
from langchain.chains.llm import LLMChain
from langchain.tools.render import ToolsRenderer, render_text_description
@property
def input_keys(self) -> List[str]:
    return ['input']