from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def embedding2bytes(embedding: Union[List[float], None]) -> Union[bytes, None]:
    """Convert embedding to bytes."""
    blob = None
    if embedding is not None:
        emb = np.array(embedding, dtype='float32')
        blob = emb.tobytes()
    return blob