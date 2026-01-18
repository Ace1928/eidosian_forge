from abc import ABC
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@property
def doc_cls(self) -> Type['BaseDoc']:
    if self.doc_index._schema is None:
        raise ValueError('doc_index expected to have non-null _schema attribute.')
    return self.doc_index._schema