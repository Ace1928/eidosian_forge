from typing import Callable, Dict, Optional, Sequence
import numpy as np
from langchain_community.document_transformers.embeddings_redundant_filter import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import (
from langchain.utils.math import cosine_similarity
def compress_documents(self, documents: Sequence[Document], query: str, callbacks: Optional[Callbacks]=None) -> Sequence[Document]:
    """Filter documents based on similarity of their embeddings to the query."""
    stateful_documents = get_stateful_documents(documents)
    embedded_documents = _get_embeddings_from_stateful_docs(self.embeddings, stateful_documents)
    embedded_query = self.embeddings.embed_query(query)
    similarity = self.similarity_fn([embedded_query], embedded_documents)[0]
    included_idxs = np.arange(len(embedded_documents))
    if self.k is not None:
        included_idxs = np.argsort(similarity)[::-1][:self.k]
    if self.similarity_threshold is not None:
        similar_enough = np.where(similarity[included_idxs] > self.similarity_threshold)
        included_idxs = included_idxs[similar_enough]
    for i in included_idxs:
        stateful_documents[i].state['query_similarity_score'] = similarity[i]
    return [stateful_documents[i] for i in included_idxs]