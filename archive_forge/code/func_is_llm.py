from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
def is_llm(llm: BaseLanguageModel) -> bool:
    """Check if the language model is a LLM.

    Args:
        llm: Language model to check.

    Returns:
        True if the language model is a BaseLLM model, False otherwise.
    """
    return isinstance(llm, BaseLLM)