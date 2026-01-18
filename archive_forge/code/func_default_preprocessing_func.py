from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
def default_preprocessing_func(text: str) -> List[str]:
    return text.split()