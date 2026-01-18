from __future__ import annotations
import datetime
import os
from typing import (
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def similarity_search_by_text(self, query: str, k: int=4, **kwargs: Any) -> List[Document]:
    """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
    content: Dict[str, Any] = {'concepts': [query]}
    if kwargs.get('search_distance'):
        content['certainty'] = kwargs.get('search_distance')
    query_obj = self._client.query.get(self._index_name, self._query_attrs)
    if kwargs.get('where_filter'):
        query_obj = query_obj.with_where(kwargs.get('where_filter'))
    if kwargs.get('tenant'):
        query_obj = query_obj.with_tenant(kwargs.get('tenant'))
    if kwargs.get('additional'):
        query_obj = query_obj.with_additional(kwargs.get('additional'))
    result = query_obj.with_near_text(content).with_limit(k).do()
    if 'errors' in result:
        raise ValueError(f'Error during query: {result['errors']}')
    docs = []
    for res in result['data']['Get'][self._index_name]:
        text = res.pop(self._text_key)
        docs.append(Document(page_content=text, metadata=res))
    return docs