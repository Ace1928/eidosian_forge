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
def from_existing_graph(cls: Type[Neo4jVector], embedding: Embeddings, node_label: str, embedding_node_property: str, text_node_properties: List[str], *, keyword_index_name: Optional[str]='keyword', index_name: str='vector', search_type: SearchType=DEFAULT_SEARCH_TYPE, retrieval_query: str='', **kwargs: Any) -> Neo4jVector:
    """
        Initialize and return a Neo4jVector instance from an existing graph.

        This method initializes a Neo4jVector instance using the provided
        parameters and the existing graph. It validates the existence of
        the indices and creates new ones if they don't exist.

        Returns:
        Neo4jVector: An instance of Neo4jVector initialized with the provided parameters
                    and existing graph.

        Example:
        >>> neo4j_vector = Neo4jVector.from_existing_graph(
        ...     embedding=my_embedding,
        ...     node_label="Document",
        ...     embedding_node_property="embedding",
        ...     text_node_properties=["title", "content"]
        ... )

        Note:
        Neo4j credentials are required in the form of `url`, `username`, and `password`,
        and optional `database` parameters passed as additional keyword arguments.
        """
    if not text_node_properties:
        raise ValueError('Parameter `text_node_properties` must not be an empty list')
    if not retrieval_query:
        retrieval_query = f"RETURN reduce(str='', k IN {text_node_properties} | str + '\\n' + k + ': ' + coalesce(node[k], '')) AS text, node {{.*, `" + embedding_node_property + '`: Null, id: Null, ' + ', '.join([f'`{prop}`: Null' for prop in text_node_properties]) + '} AS metadata, score'
    store = cls(embedding=embedding, index_name=index_name, keyword_index_name=keyword_index_name, search_type=search_type, retrieval_query=retrieval_query, node_label=node_label, embedding_node_property=embedding_node_property, **kwargs)
    embedding_dimension, index_type = store.retrieve_existing_index()
    if index_type == 'RELATIONSHIP':
        raise ValueError('`from_existing_graph` method does not support  existing relationship vector index. Please use `from_existing_relationship_index` method')
    if not embedding_dimension:
        store.create_new_index()
    elif not store.embedding_dimension == embedding_dimension:
        raise ValueError(f'Index with name {store.index_name} already exists.The provided embedding function and vector index dimensions do not match.\nEmbedding function dimension: {store.embedding_dimension}\nVector index dimension: {embedding_dimension}')
    if search_type == SearchType.HYBRID:
        fts_node_label = store.retrieve_existing_fts_index(text_node_properties)
        if not fts_node_label:
            store.create_new_keyword_index(text_node_properties)
        elif not fts_node_label == store.node_label:
            raise ValueError("Vector and keyword index don't index the same node label")
    while True:
        fetch_query = f"MATCH (n:`{node_label}`) WHERE n.{embedding_node_property} IS null AND any(k in $props WHERE n[k] IS NOT null) RETURN elementId(n) AS id, reduce(str='',k IN $props | str + '\\n' + k + ':' + coalesce(n[k], '')) AS text LIMIT 1000"
        data = store.query(fetch_query, params={'props': text_node_properties})
        text_embeddings = embedding.embed_documents([el['text'] for el in data])
        params = {'data': [{'id': el['id'], 'embedding': embedding} for el, embedding in zip(data, text_embeddings)]}
        store.query(f"UNWIND $data AS row MATCH (n:`{node_label}`) WHERE elementId(n) = row.id CALL db.create.setVectorProperty(n, '{embedding_node_property}', row.embedding) YIELD node RETURN count(*)", params=params)
        if len(data) < 1000:
            break
    return store