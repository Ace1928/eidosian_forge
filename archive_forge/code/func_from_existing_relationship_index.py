from __future__ import annotations
import enum
import logging
import os
from hashlib import md5
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
@classmethod
def from_existing_relationship_index(cls: Type[Neo4jVector], embedding: Embeddings, index_name: str, search_type: SearchType=DEFAULT_SEARCH_TYPE, **kwargs: Any) -> Neo4jVector:
    """
        Get instance of an existing Neo4j relationship vector index.
        This method will return the instance of the store without
        inserting any new embeddings.
        Neo4j credentials are required in the form of `url`, `username`,
        and `password` and optional `database` parameters along with
        the `index_name` definition.
        """
    if search_type == SearchType.HYBRID:
        raise ValueError('Hybrid search is not supported in combination with relationship vector index')
    store = cls(embedding=embedding, index_name=index_name, **kwargs)
    embedding_dimension, index_type = store.retrieve_existing_index()
    if not embedding_dimension:
        raise ValueError('The specified vector index name does not exist. Make sure to check if you spelled it correctly')
    if index_type == 'NODE':
        raise ValueError('Node vector index is not supported with `from_existing_relationship_index` method. Please use the `from_existing_index` method.')
    if not store.embedding_dimension == embedding_dimension:
        raise ValueError(f'The provided embedding function and vector index dimensions do not match.\nEmbedding function dimension: {store.embedding_dimension}\nVector index dimension: {embedding_dimension}')
    return store