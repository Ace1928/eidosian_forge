import time
from typing import Any, Dict, List, Optional, cast
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
def import_tiktoken() -> Any:
    """Import tiktoken for counting tokens for OpenAI models."""
    try:
        import tiktoken
    except ImportError:
        raise ImportError('To use the ChatOpenAI model with Infino callback manager, you need to have the `tiktoken` python package installed.Please install it with `pip install tiktoken`')
    return tiktoken