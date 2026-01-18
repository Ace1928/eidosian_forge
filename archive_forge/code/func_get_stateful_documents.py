from typing import Any, Callable, List, Sequence
import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utils.math import cosine_similarity
def get_stateful_documents(documents: Sequence[Document]) -> Sequence[_DocumentWithState]:
    """Convert a list of documents to a list of documents with state.

    Args:
        documents: The documents to convert.

    Returns:
        A list of documents with state.
    """
    return [_DocumentWithState.from_document(doc) for doc in documents]