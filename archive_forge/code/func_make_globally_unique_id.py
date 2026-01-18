from __future__ import annotations
import logging # isort:skip
import datetime as dt
import uuid
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any
import numpy as np
from ..core.types import ID
from ..settings import settings
from .strings import format_docstring
def make_globally_unique_id() -> ID:
    """ Return a globally unique UUID.

    Some situations, e.g. id'ing dynamically created Divs in HTML documents,
    always require globally unique IDs.

    Returns:
        str

    """
    return ID(str(uuid.uuid4()))