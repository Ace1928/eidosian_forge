import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
def combined_text(item: 'ResultItem') -> str:
    """Combine a ResultItem title and excerpt into a single string.

    Args:
        item: the ResultItem of a Kendra search.

    Returns:
        A combined text of the title and excerpt of the given item.

    """
    text = ''
    title = item.get_title()
    if title:
        text += f'Document Title: {title}\n'
    excerpt = clean_excerpt(item.get_excerpt())
    if excerpt:
        text += f'Document Excerpt: \n{excerpt}\n'
    return text