from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
import sqlalchemy as sa
from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.sql_database import SQLDatabase
@staticmethod
def metadata_default_mapper(row: sa.RowMapping, column_names: Optional[List[str]]=None) -> Dict[str, Any]:
    """
        A reasonable default function to convert a record into a "metadata" dictionary.
        """
    if column_names is None:
        return {}
    metadata: Dict[str, Any] = {}
    for column, value in row.items():
        if column in column_names:
            metadata[column] = value
    return metadata