from copy import deepcopy
from typing import Iterable, Iterator, List
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, BaseMessage
def map_ai_messages(chat_sessions: Iterable[ChatSession], sender: str) -> Iterator[ChatSession]:
    """Convert messages from the specified 'sender' to AI messages.

    This is useful for fine-tuning the AI to adapt to your voice.
    """
    for chat_session in chat_sessions:
        yield map_ai_messages_in_session(chat_session, sender)