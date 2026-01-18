from __future__ import annotations
import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
def get_named_result(connection: Any, query: str) -> List[dict[str, Any]]:
    """
    Get a named result from a query.
    Args:
        connection: The connection to the database
        query: The query to execute

    Returns:
        List[dict[str, Any]]: The result of the query
    """
    cursor = connection.cursor()
    cursor.execute(query)
    columns = cursor.description
    result = []
    for value in cursor.fetchall():
        r = {}
        for idx, datum in enumerate(value):
            k = columns[idx][0]
            r[k] = datum
        result.append(r)
    debug_output(result)
    cursor.close()
    return result