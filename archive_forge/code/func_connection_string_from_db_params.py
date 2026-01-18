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
@classmethod
def connection_string_from_db_params(cls, driver: str, host: str, port: int, database: str, user: str, password: str) -> str:
    """Return connection string from database parameters."""
    return f'postgresql+{driver}://{user}:{password}@{host}:{port}/{database}'