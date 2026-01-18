from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
import sqlalchemy as sa
from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.sql_database import SQLDatabase
@staticmethod
def page_content_default_mapper(row: sa.RowMapping, column_names: Optional[List[str]]=None) -> str:
    """
        A reasonable default function to convert a record into a "page content" string.
        """
    if column_names is None:
        column_names = list(row.keys())
    return '\n'.join((f'{column}: {value}' for column, value in row.items() if column in column_names))