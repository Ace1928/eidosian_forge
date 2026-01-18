from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_community.llms.anthropic import _AnthropicCommon
def convert_messages_to_prompt_anthropic(messages: List[BaseMessage], *, human_prompt: str='\n\nHuman:', ai_prompt: str='\n\nAssistant:') -> str:
    """Format a list of messages into a full prompt for the Anthropic model
    Args:
        messages (List[BaseMessage]): List of BaseMessage to combine.
        human_prompt (str, optional): Human prompt tag. Defaults to "

Human:".
        ai_prompt (str, optional): AI prompt tag. Defaults to "

Assistant:".
    Returns:
        str: Combined string with necessary human_prompt and ai_prompt tags.
    """
    messages = messages.copy()
    if not isinstance(messages[-1], AIMessage):
        messages.append(AIMessage(content=''))
    text = ''.join((_convert_one_message_to_text(message, human_prompt, ai_prompt) for message in messages))
    return text.rstrip()