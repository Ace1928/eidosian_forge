from __future__ import annotations
import enum
import logging
import uuid
from datetime import timedelta
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
@classmethod
def get_service_url(cls, kwargs: Dict[str, Any]) -> str:
    service_url: str = get_from_dict_or_env(data=kwargs, key='service_url', env_key='TIMESCALE_SERVICE_URL')
    if not service_url:
        raise ValueError('Postgres connection string is requiredEither pass it as a parameteror set the TIMESCALE_SERVICE_URL environment variable.')
    return service_url