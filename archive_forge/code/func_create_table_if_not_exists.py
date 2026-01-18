from __future__ import annotations
import logging
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type
from sqlalchemy import REAL, Column, String, Table, create_engine, insert, text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, TEXT
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def create_table_if_not_exists(self) -> None:
    Table(self.collection_name, Base.metadata, Column('id', TEXT, primary_key=True, default=uuid.uuid4), Column('embedding', ARRAY(REAL)), Column('document', String, nullable=True), Column('metadata', JSON, nullable=True), extend_existing=True)
    with self.engine.connect() as conn:
        with conn.begin():
            Base.metadata.create_all(conn)
            index_name = f'{self.collection_name}_embedding_idx'
            index_query = text(f"\n                    SELECT 1\n                    FROM pg_indexes\n                    WHERE indexname = '{index_name}';\n                ")
            result = conn.execute(index_query).scalar()
            if not result:
                index_statement = text(f'\n                        CREATE INDEX {index_name}\n                        ON {self.collection_name} USING ann(embedding)\n                        WITH (\n                            "dim" = {self.embedding_dimension},\n                            "hnsw_m" = 100\n                        );\n                    ')
                conn.execute(index_statement)