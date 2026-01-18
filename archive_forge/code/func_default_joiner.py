from typing import Any, Callable, Iterator, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def default_joiner(docs: List[Tuple[str, Any]]) -> str:
    """Default joiner for content columns."""
    return '\n'.join([doc[1] for doc in docs])