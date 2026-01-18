import json
import logging
from typing import Any, Callable, Iterable, List, Optional, Tuple
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def get_input_files(self, metadata_filter: Optional[str]=None, filepath_globpattern: Optional[str]=None) -> list:
    """List files indexed by the Vector Store."""
    return self.client.get_input_files(metadata_filter, filepath_globpattern)