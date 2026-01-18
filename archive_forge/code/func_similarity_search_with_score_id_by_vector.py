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
def similarity_search_with_score_id_by_vector(self, embedding: List[float], k: int=4, filter: Optional[Dict[str, Any]]=None) -> List[Tuple[Document, float, str]]:
    """Return docs most similar to embedding vector with score and id.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query vector.
        """
    self.astra_env.ensure_db_setup()
    metadata_parameter = self._filter_to_metadata(filter)
    hits = list(self.collection.paginated_find(filter=metadata_parameter, sort={'$vector': embedding}, options={'limit': k, 'includeSimilarity': True}, projection={'_id': 1, 'content': 1, 'metadata': 1}))
    return [(Document(page_content=hit['content'], metadata=hit['metadata']), hit['$similarity'], hit['_id']) for hit in hits]