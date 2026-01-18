import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Extra
from langchain_community.chat_models.anthropic import (
from langchain_community.chat_models.meta import convert_messages_to_prompt_llama
from langchain_community.llms.bedrock import BedrockBase
from langchain_community.utilities.anthropic import (
def convert_messages_to_prompt_mistral(messages: List[BaseMessage]) -> str:
    """Convert a list of messages to a prompt for mistral."""
    return '\n'.join([_convert_one_message_to_text_mistral(message) for message in messages])