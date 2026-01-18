from __future__ import annotations
import base64
import json
import logging
import uuid
from typing import (
import numpy as np
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
def semantic_hybrid_search_with_score_and_rerank(self, query: str, k: int=4, filters: Optional[str]=None) -> List[Tuple[Document, float, float]]:
    """Return docs most similar to query with an hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
    from azure.search.documents.models import VectorizedQuery
    results = self.client.search(search_text=query, vector_queries=[VectorizedQuery(vector=np.array(self.embed_query(query), dtype=np.float32).tolist(), k_nearest_neighbors=k, fields=FIELDS_CONTENT_VECTOR)], filter=filters, query_type='semantic', semantic_configuration_name=self.semantic_configuration_name, query_caption='extractive', query_answer='extractive', top=k)
    semantic_answers = results.get_answers() or []
    semantic_answers_dict: Dict = {}
    for semantic_answer in semantic_answers:
        semantic_answers_dict[semantic_answer.key] = {'text': semantic_answer.text, 'highlights': semantic_answer.highlights}
    docs = [(Document(page_content=result.pop(FIELDS_CONTENT), metadata={**(json.loads(result[FIELDS_METADATA]) if FIELDS_METADATA in result else {k: v for k, v in result.items() if k != FIELDS_CONTENT_VECTOR}), **{'captions': {'text': result.get('@search.captions', [{}])[0].text, 'highlights': result.get('@search.captions', [{}])[0].highlights} if result.get('@search.captions') else {}, 'answers': semantic_answers_dict.get(result.get(FIELDS_ID, ''), '')}}), float(result['@search.score']), float(result['@search.reranker_score'])) for result in results]
    return docs