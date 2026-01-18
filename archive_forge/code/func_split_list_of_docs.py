from __future__ import annotations
from typing import Any, Callable, List, Optional, Protocol, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
def split_list_of_docs(docs: List[Document], length_func: Callable, token_max: int, **kwargs: Any) -> List[List[Document]]:
    """Split Documents into subsets that each meet a cumulative length constraint.

    Args:
        docs: The full list of Documents.
        length_func: Function for computing the cumulative length of a set of Documents.
        token_max: The maximum cumulative length of any subset of Documents.
        **kwargs: Arbitrary additional keyword params to pass to each call of the
            length_func.

    Returns:
        A List[List[Document]].
    """
    new_result_doc_list = []
    _sub_result_docs = []
    for doc in docs:
        _sub_result_docs.append(doc)
        _num_tokens = length_func(_sub_result_docs, **kwargs)
        if _num_tokens > token_max:
            if len(_sub_result_docs) == 1:
                raise ValueError('A single document was longer than the context length, we cannot handle this.')
            new_result_doc_list.append(_sub_result_docs[:-1])
            _sub_result_docs = _sub_result_docs[-1:]
    new_result_doc_list.append(_sub_result_docs)
    return new_result_doc_list