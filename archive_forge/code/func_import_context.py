import os
from typing import Any, Dict, List
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
def import_context() -> Any:
    """Import the `getcontext` package."""
    try:
        import getcontext
        from getcontext.generated.models import Conversation, Message, MessageRole, Rating
        from getcontext.token import Credential
    except ImportError:
        raise ImportError('To use the context callback manager you need to have the `getcontext` python package installed (version >=0.3.0). Please install it with `pip install --upgrade python-context`')
    return (getcontext, Credential, Conversation, Message, MessageRole, Rating)