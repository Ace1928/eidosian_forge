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
def combine_queries(input_queries: List[Tuple[str, Dict[str, Any]]], operator: str) -> Tuple[str, Dict[str, Any]]:
    """Combine multiple queries with an operator."""
    combined_query: str = ''
    combined_params: Dict = {}
    param_counter: Dict = {}
    for query, params in input_queries:
        new_query = query
        for param, value in params.items():
            if param in param_counter:
                param_counter[param] += 1
            else:
                param_counter[param] = 1
            new_param_name = f'{param}_{param_counter[param]}'
            new_query = new_query.replace(f'${param}', f'${new_param_name}')
            combined_params[new_param_name] = value
        if combined_query:
            combined_query += f' {operator} '
        combined_query += f'({new_query})'
    return (combined_query, combined_params)