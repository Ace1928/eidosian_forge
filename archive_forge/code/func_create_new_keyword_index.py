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
def create_new_keyword_index(self, text_node_properties: List[str]=[]) -> None:
    """
        This method constructs a Cypher query and executes it
        to create a new full text index in Neo4j.
        """
    node_props = text_node_properties or [self.text_node_property]
    fts_index_query = f'CREATE FULLTEXT INDEX {self.keyword_index_name} FOR (n:`{self.node_label}`) ON EACH [{', '.join(['n.`' + el + '`' for el in node_props])}]'
    self.query(fts_index_query)