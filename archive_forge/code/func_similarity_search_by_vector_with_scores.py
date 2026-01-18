from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
def similarity_search_by_vector_with_scores(self, embedding: List[float], embedder_name: Optional[str]='default', k: int=4, filter: Optional[Dict[str, Any]]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
    """Return meilisearch documents most similar to embedding vector.

        Args:
            embedding (List[float]): Embedding to look up similar documents.
            embedder_name: Name of the embedder to be used. Defaults to "default".
            k (int): Number of documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None.

        Returns:
            List[Document]: List of Documents most similar to the query
                vector and score for each.
        """
    docs = []
    results = self._client.index(str(self._index_name)).search('', {'vector': embedding, 'hybrid': {'semanticRatio': 1.0, 'embedder': embedder_name}, 'limit': k, 'filter': filter})
    for result in results['hits']:
        metadata = result[self._metadata_key]
        if self._text_key in metadata:
            text = metadata.pop(self._text_key)
            semantic_score = result['_semanticScore']
            docs.append((Document(page_content=text, metadata=metadata), semantic_score))
    return docs