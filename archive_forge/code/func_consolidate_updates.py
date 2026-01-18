from __future__ import annotations
import pickle
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def consolidate_updates(self, **kwargs: Any) -> None:
    self.vector_index = self.vector_index.consolidate_updates(**kwargs)