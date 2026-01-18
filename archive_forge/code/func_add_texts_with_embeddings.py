from __future__ import annotations
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utils.google import get_client_info
from langchain_community.vectorstores.utils import (
def add_texts_with_embeddings(self, texts: List[str], embs: List[List[float]], metadatas: Optional[List[dict]]=None, **kwargs: Any) -> List[str]:
    """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: List of strings to add to the vectorstore.
            embs: List of lists of floats with text embeddings for texts.
            metadatas: Optional list of metadata associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
    ids = [uuid.uuid4().hex for _ in texts]
    values_dict: Dict[str, List[Any]] = {self.content_field: texts, self.doc_id_field: ids}
    if not metadatas:
        metadatas = []
    len_diff = len(ids) - len(metadatas)
    add_meta = [None for _ in range(0, len_diff)]
    metadatas = [m if m is not None else {} for m in metadatas + add_meta]
    values_dict[self.metadata_field] = metadatas
    values_dict[self.text_embedding_field] = embs
    self._persist(values_dict)
    return ids