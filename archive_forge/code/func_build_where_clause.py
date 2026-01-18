from __future__ import annotations
import json
import re
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from sqlalchemy.pool import QueuePool
from langchain_community.vectorstores.utils import DistanceStrategy
def build_where_clause(where_clause_values: List[Any], sub_filter: dict, prefix_args: Optional[List[str]]=None) -> None:
    prefix_args = prefix_args or []
    for key in sub_filter.keys():
        if isinstance(sub_filter[key], dict):
            build_where_clause(where_clause_values, sub_filter[key], prefix_args + [key])
        else:
            arguments.append('JSON_EXTRACT_JSON({}, {}) = %s'.format(self.metadata_field, ', '.join(['%s'] * (len(prefix_args) + 1))))
            where_clause_values += prefix_args + [key]
            where_clause_values.append(json.dumps(sub_filter[key]))