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
def create_hnsw_extension(self) -> None:
    try:
        with Session(self._conn) as session:
            statement = sqlalchemy.text('CREATE EXTENSION IF NOT EXISTS embedding')
            session.execute(statement)
            session.commit()
    except Exception as e:
        self.logger.exception(e)