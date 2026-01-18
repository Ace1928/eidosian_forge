from __future__ import annotations
import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
@property
def metadata_column(self) -> str:
    return self.config.column_map['metadata']