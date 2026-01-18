from __future__ import annotations
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import sqlalchemy
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Session, relationship
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
@classmethod
def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
    connection_string: str = get_from_dict_or_env(data=kwargs, key='connection_string', env_key='POSTGRES_CONNECTION_STRING')
    if not connection_string:
        raise ValueError('Postgres connection string is requiredEither pass it as a parameteror set the POSTGRES_CONNECTION_STRING environment variable.')
    return connection_string