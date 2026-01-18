import json
import logging
from typing import Any, Callable, Iterable, List, Optional, Tuple
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def get_vectorstore_statistics(self) -> dict:
    """Fetch basic statistics about the Vector Store."""
    return self.client.get_vectorstore_statistics()