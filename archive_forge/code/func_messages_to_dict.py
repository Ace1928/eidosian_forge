from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.interactive_env import is_interactive_env
def messages_to_dict(messages: Sequence[BaseMessage]) -> List[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as BaseMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [message_to_dict(m) for m in messages]