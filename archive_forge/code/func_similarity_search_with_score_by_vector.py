from __future__ import annotations
import asyncio
import enum
import json
import logging
import struct
import uuid
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def similarity_search_with_score_by_vector(self, embedding: List[float], k: int=4, filter: Optional[dict]=None) -> List[Tuple[Document, float]]:
    resp: Dict = self.__query_collection(embedding, k, filter)
    records: OrderedDict = resp['records']
    results = list(zip(*list(records.values())))
    return self._results_to_docs_and_scores(results)