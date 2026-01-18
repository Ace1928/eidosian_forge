import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
def clean_excerpt(excerpt: str) -> str:
    """Clean an excerpt from Kendra.

    Args:
        excerpt: The excerpt to clean.

    Returns:
        The cleaned excerpt.

    """
    if not excerpt:
        return excerpt
    res = re.sub('\\s+', ' ', excerpt).replace('...', '')
    return res