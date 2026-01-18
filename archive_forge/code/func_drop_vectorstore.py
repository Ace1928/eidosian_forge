import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def drop_vectorstore(self) -> None:
    """
        Drop the Vector Store from the TiDB database.
        """
    self._tidb.drop_table()