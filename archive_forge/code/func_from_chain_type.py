from __future__ import annotations
import inspect
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
@classmethod
def from_chain_type(cls, llm: BaseLanguageModel, chain_type: str='stuff', chain_type_kwargs: Optional[dict]=None, **kwargs: Any) -> BaseRetrievalQA:
    """Load chain from chain type."""
    _chain_type_kwargs = chain_type_kwargs or {}
    combine_documents_chain = load_qa_chain(llm, chain_type=chain_type, **_chain_type_kwargs)
    return cls(combine_documents_chain=combine_documents_chain, **kwargs)