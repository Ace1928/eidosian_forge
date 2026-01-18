from __future__ import annotations
import contextlib
import enum
import logging
import uuid
from typing import (
import numpy as np
import sqlalchemy
from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session
from sqlalchemy.sql import quoted_name
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def get_embedding_store(distance_strategy: DistanceStrategy, collection_name: str) -> Any:
    """Get the embedding store class."""
    embedding_type = None
    if distance_strategy == DistanceStrategy.HAMMING:
        embedding_type = sqlalchemy.INTEGER
    else:
        embedding_type = sqlalchemy.REAL
    DynamicBase = declarative_base(class_registry=dict())

    class EmbeddingStore(DynamicBase, BaseEmbeddingStore):
        __tablename__ = collection_name
        uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        __table_args__ = {'extend_existing': True}
        document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
        cmetadata = sqlalchemy.Column(JSON, nullable=True)
        custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)
        embedding = sqlalchemy.Column(sqlalchemy.ARRAY(embedding_type))
    return EmbeddingStore