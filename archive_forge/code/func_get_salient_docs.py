import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
def get_salient_docs(self, query: str) -> Dict[int, Tuple[Document, float]]:
    """Return documents that are salient to the query."""
    docs_and_scores: List[Tuple[Document, float]]
    docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(query, **self.search_kwargs)
    results = {}
    for fetched_doc, relevance in docs_and_scores:
        if 'buffer_idx' in fetched_doc.metadata:
            buffer_idx = fetched_doc.metadata['buffer_idx']
            doc = self.memory_stream[buffer_idx]
            results[buffer_idx] = (doc, relevance)
    return results