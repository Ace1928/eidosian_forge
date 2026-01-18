from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
@property
def ignore_retry(self) -> bool:
    """Whether to ignore retry callbacks."""
    return False