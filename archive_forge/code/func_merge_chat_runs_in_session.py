from copy import deepcopy
from typing import Iterable, Iterator, List
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, BaseMessage
def merge_chat_runs_in_session(chat_session: ChatSession, delimiter: str='\n\n') -> ChatSession:
    """Merge chat runs together in a chat session.

    A chat run is a sequence of messages from the same sender.

    Args:
        chat_session: A chat session.

    Returns:
        A chat session with merged chat runs.
    """
    messages: List[BaseMessage] = []
    for message in chat_session['messages']:
        if not isinstance(message.content, str):
            raise ValueError(f'Chat Loaders only support messages with content type string, got {message.content}')
        if not messages:
            messages.append(deepcopy(message))
        elif isinstance(message, type(messages[-1])) and messages[-1].additional_kwargs.get('sender') is not None and (messages[-1].additional_kwargs['sender'] == message.additional_kwargs.get('sender')):
            if not isinstance(messages[-1].content, str):
                raise ValueError(f'Chat Loaders only support messages with content type string, got {messages[-1].content}')
            messages[-1].content = (messages[-1].content + delimiter + message.content).strip()
            messages[-1].additional_kwargs.get('events', []).extend(message.additional_kwargs.get('events') or [])
        else:
            messages.append(deepcopy(message))
    return ChatSession(messages=messages)