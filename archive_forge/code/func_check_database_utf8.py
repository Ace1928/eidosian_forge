from __future__ import annotations
import json
import logging
import uuid
import warnings
from itertools import repeat
from typing import (
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.document import Document
def check_database_utf8(self) -> bool:
    """
        Helper function: Test the database is UTF-8 encoded
        """
    cursor = self._connection.cursor()
    query = 'SELECT pg_encoding_to_char(encoding)         FROM pg_database         WHERE datname = current_database();'
    cursor.execute(query)
    encoding = cursor.fetchone()[0]
    cursor.close()
    if encoding.lower() == 'utf8' or encoding.lower() == 'utf-8':
        return True
    else:
        raise Exception(f"Database            '{self.connection_string.split('/')[-1]}' encoding is not UTF-8")