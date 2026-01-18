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
def create_new_index(self) -> None:
    """
        This method constructs a Cypher query and executes it
        to create a new vector index in Neo4j.
        """
    index_query = 'CALL db.index.vector.createNodeIndex($index_name,$node_label,$embedding_node_property,toInteger($embedding_dimension),$similarity_metric )'
    parameters = {'index_name': self.index_name, 'node_label': self.node_label, 'embedding_node_property': self.embedding_node_property, 'embedding_dimension': self.embedding_dimension, 'similarity_metric': DISTANCE_MAPPING[self._distance_strategy]}
    self.query(index_query, params=parameters)