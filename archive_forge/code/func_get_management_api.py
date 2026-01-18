from __future__ import annotations
from pydantic import BaseModel as _BaseModel
from typing import TYPE_CHECKING, Optional
@classmethod
def get_management_api(cls) -> 'AZManagementAPI':
    """
        Returns the management api
        """
    from ..utils.lazy import get_az_mtg_api
    return get_az_mtg_api()