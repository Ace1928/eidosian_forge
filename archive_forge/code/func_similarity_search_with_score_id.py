from __future__ import annotations
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def similarity_search_with_score_id(self, query: str, k: int=4, filter: Optional[Dict[str, Any]]=None) -> List[Tuple[Document, float, str]]:
    """Return docs most similar to the query with score and id.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query.
        """
    embedding_vector = self.embedding.embed_query(query)
    return self.similarity_search_with_score_id_by_vector(embedding=embedding_vector, k=k, filter=filter)