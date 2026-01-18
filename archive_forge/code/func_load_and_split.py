from __future__ import annotations
from pathlib import Path
from typing import (
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser, BaseLoader
from langchain_community.document_loaders.blob_loaders import (
from langchain_community.document_loaders.parsers.registry import get_parser
def load_and_split(self, text_splitter: Optional[TextSplitter]=None) -> List[Document]:
    """Load all documents and split them into sentences."""
    raise NotImplementedError('Loading and splitting is not yet implemented for generic loaders. When they will be implemented they will be added via the initializer. This method should not be used going forward.')