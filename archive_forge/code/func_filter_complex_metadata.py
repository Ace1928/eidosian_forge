from enum import Enum
from typing import List, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_community.utils.math import cosine_similarity
def filter_complex_metadata(documents: List[Document], *, allowed_types: Tuple[Type, ...]=(str, bool, int, float)) -> List[Document]:
    """Filter out metadata types that are not supported for a vector store."""
    updated_documents = []
    for document in documents:
        filtered_metadata = {}
        for key, value in document.metadata.items():
            if not isinstance(value, allowed_types):
                continue
            filtered_metadata[key] = value
        document.metadata = filtered_metadata
        updated_documents.append(document)
    return updated_documents