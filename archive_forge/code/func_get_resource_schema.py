from __future__ import annotations
from pydantic import BaseModel as _BaseModel
from typing import TYPE_CHECKING, Optional
@classmethod
def get_resource_schema(cls, name: str) -> 'AZResourceSchema':
    """
        Returns the resource schema
        """
    from ..utils.lazy import get_az_resource_schema
    return get_az_resource_schema(name)