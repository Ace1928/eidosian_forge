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
def date_to_range_filter(self, **kwargs: Any) -> Any:
    constructor_args = {key: kwargs[key] for key in ['start_date', 'end_date', 'time_delta', 'start_inclusive', 'end_inclusive'] if key in kwargs}
    if not constructor_args or len(constructor_args) == 0:
        return None
    try:
        from timescale_vector import client
    except ImportError:
        raise ImportError('Could not import timescale_vector python package. Please install it with `pip install timescale-vector`.')
    return client.UUIDTimeRange(**constructor_args)