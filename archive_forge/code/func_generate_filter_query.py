import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def generate_filter_query() -> str:
    if search_filter is None:
        return ''
    filter_clause = ' AND '.join([create_filter(md_key, md_value) for md_key, md_value in search_filter.items()])
    return filter_clause