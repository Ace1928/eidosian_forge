import html
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar
from triad import ParamDict, SerializableRLock, assert_or_throw
from .._utils.registry import fugue_plugin
from ..exceptions import FugueDatasetEmptyError
def reset_metadata(self, metadata: Any) -> None:
    """Reset metadata"""
    self._metadata = ParamDict(metadata) if metadata is not None else None