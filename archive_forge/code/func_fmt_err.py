from __future__ import annotations
import base64
import json
import logging
import uuid
from typing import (
import numpy as np
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
def fmt_err(x: str) -> str:
    return f"{x} current type: '{fields_types.get(x, 'MISSING')}'. It has to be '{mandatory_fields.get(x)}' or you can point to a different '{mandatory_fields.get(x)}' field name by using the env variable 'AZURESEARCH_FIELDS_{x.upper()}'"