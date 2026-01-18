from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.vectorstores import VectorStore
def sorted_values(values: Dict[str, str]) -> List[Any]:
    """Return a list of values in dict sorted by key."""
    return [values[val] for val in sorted(values)]