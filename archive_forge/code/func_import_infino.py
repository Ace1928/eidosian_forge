import time
from typing import Any, Dict, List, Optional, cast
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
def import_infino() -> Any:
    """Import the infino client."""
    try:
        from infinopy import InfinoClient
    except ImportError:
        raise ImportError('To use the Infino callbacks manager you need to have the `infinopy` python package installed.Please install it with `pip install infinopy`')
    return InfinoClient()