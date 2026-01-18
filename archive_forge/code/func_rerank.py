from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.utils import get_from_dict_or_env
def rerank(self, documents: Sequence[Union[str, Document, dict]], query: str, *, model: Optional[str]=None, top_n: Optional[int]=-1, max_chunks_per_doc: Optional[int]=None) -> List[Dict[str, Any]]:
    """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
            max_chunks_per_doc : The maximum number of chunks derived from a document.
        """
    if len(documents) == 0:
        return []
    docs = [doc.page_content if isinstance(doc, Document) else doc for doc in documents]
    model = model or self.model
    top_n = top_n if top_n is None or top_n > 0 else self.top_n
    results = self.client.rerank(query=query, documents=docs, model=model, top_n=top_n, max_chunks_per_doc=max_chunks_per_doc)
    if hasattr(results, 'results'):
        results = getattr(results, 'results')
    result_dicts = []
    for res in results:
        result_dicts.append({'index': res.index, 'relevance_score': res.relevance_score})
    return result_dicts