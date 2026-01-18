from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing_extensions import TYPE_CHECKING, Literal
from langchain_community.vectorstores.redis.constants import REDIS_VECTOR_DTYPE_MAP
def add_vector_field(self, vector_field: Dict[str, Any]) -> None:
    if self.vector is None:
        self.vector = []
    if vector_field['algorithm'] == 'FLAT':
        self.vector.append(FlatVectorField(**vector_field))
    elif vector_field['algorithm'] == 'HNSW':
        self.vector.append(HNSWVectorField(**vector_field))
    else:
        raise ValueError(f'algorithm must be either FLAT or HNSW. Got {vector_field['algorithm']}')