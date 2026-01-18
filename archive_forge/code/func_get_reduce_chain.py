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
@root_validator(pre=True)
def get_reduce_chain(cls, values: Dict) -> Dict:
    """For backwards compatibility."""
    if 'combine_document_chain' in values:
        if 'reduce_documents_chain' in values:
            raise ValueError('Both `reduce_documents_chain` and `combine_document_chain` cannot be provided at the same time. `combine_document_chain` is deprecated, please only provide `reduce_documents_chain`')
        combine_chain = values['combine_document_chain']
        collapse_chain = values.get('collapse_document_chain')
        reduce_chain = ReduceDocumentsChain(combine_documents_chain=combine_chain, collapse_documents_chain=collapse_chain)
        values['reduce_documents_chain'] = reduce_chain
        del values['combine_document_chain']
        if 'collapse_document_chain' in values:
            del values['collapse_document_chain']
    return values