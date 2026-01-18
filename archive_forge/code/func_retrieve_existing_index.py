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
def retrieve_existing_index(self) -> Tuple[Optional[int], Optional[str]]:
    """
        Check if the vector index exists in the Neo4j database
        and returns its embedding dimension.

        This method queries the Neo4j database for existing indexes
        and attempts to retrieve the dimension of the vector index
        with the specified name. If the index exists, its dimension is returned.
        If the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.
        """
    index_information = self.query("SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options WHERE type = 'VECTOR' AND (name = $index_name OR (labelsOrTypes[0] = $node_label AND properties[0] = $embedding_node_property)) RETURN name, entityType, labelsOrTypes, properties, options ", params={'index_name': self.index_name, 'node_label': self.node_label, 'embedding_node_property': self.embedding_node_property})
    index_information = sort_by_index_name(index_information, self.index_name)
    try:
        self.index_name = index_information[0]['name']
        self.node_label = index_information[0]['labelsOrTypes'][0]
        self.embedding_node_property = index_information[0]['properties'][0]
        self._index_type = index_information[0]['entityType']
        embedding_dimension = index_information[0]['options']['indexConfig']['vector.dimensions']
        return (embedding_dimension, index_information[0]['entityType'])
    except IndexError:
        return (None, None)