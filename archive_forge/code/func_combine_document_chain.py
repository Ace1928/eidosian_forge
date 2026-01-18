from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Type
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import create_model
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.llm import LLMChain
@property
def combine_document_chain(self) -> BaseCombineDocumentsChain:
    """Kept for backward compatibility."""
    if isinstance(self.reduce_documents_chain, ReduceDocumentsChain):
        return self.reduce_documents_chain.combine_documents_chain
    else:
        raise ValueError(f'`reduce_documents_chain` is of type {type(self.reduce_documents_chain)} so it does not have this attribute.')