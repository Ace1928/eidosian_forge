from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False