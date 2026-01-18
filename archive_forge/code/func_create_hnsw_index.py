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
def create_hnsw_index(self, max_elements: int=10000, dims: int=ADA_TOKEN_COUNT, m: int=8, ef_construction: int=16, ef_search: int=16) -> None:
    create_index_query = sqlalchemy.text('CREATE INDEX IF NOT EXISTS langchain_pg_embedding_idx ON langchain_pg_embedding USING hnsw (embedding) WITH (maxelements = {}, dims = {}, m = {}, efconstruction = {}, efsearch = {});'.format(max_elements, dims, m, ef_construction, ef_search))
    try:
        with Session(self._conn) as session:
            session.execute(create_index_query)
            session.commit()
        print('HNSW extension and index created successfully.')
    except Exception as e:
        print(f'Failed to create HNSW extension or index: {e}')