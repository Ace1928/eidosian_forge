from __future__ import annotations
import hashlib
import inspect
import json
import logging
import uuid
import warnings
from abc import ABC
from datetime import timedelta
from enum import Enum
from functools import lru_cache, wraps
from typing import (
from sqlalchemy import Column, Integer, String, create_engine, delete, select
from sqlalchemy.engine import Row
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from langchain_community.vectorstores.azure_cosmos_db import (
from langchain_core._api.deprecation import deprecated
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM, aget_prompts, get_prompts
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils import get_from_env
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from langchain_community.vectorstores.redis import Redis as RedisVectorstore
def lookup_with_id(self, prompt: str, llm_string: str) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
    """
        Look up based on prompt and llm_string.
        If there are hits, return (document_id, cached_entry) for the top hit
        """
    self.astra_env.ensure_db_setup()
    prompt_embedding: List[float] = self._get_embedding(text=prompt)
    llm_string_hash = _hash(llm_string)
    hit = self.collection.vector_find_one(vector=prompt_embedding, filter={'llm_string_hash': llm_string_hash}, fields=['body_blob', '_id'], include_similarity=True)
    if hit is None or hit['$similarity'] < self.similarity_threshold:
        return None
    else:
        generations = _loads_generations(hit['body_blob'])
        if generations is not None:
            return (hit['_id'], generations)
        else:
            return None