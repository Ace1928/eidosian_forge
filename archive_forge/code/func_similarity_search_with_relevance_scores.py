import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def similarity_search_with_relevance_scores(self, query: str, k: int=4, search_filter: Optional[dict]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
    """Perform similarity retrieval based on text with scores.
        Args:
            query: Vectorize text for retrieval.，should not be empty.
            k: top n.
            search_filter: Additional filtering conditions.
        Returns:
            document_list: List of documents.
        """
    embedding: List[float] = self.embedding.embed_query(query)
    return self.create_results_with_score(self.inner_embedding_query(embedding=embedding, search_filter=search_filter, k=k))