from __future__ import annotations
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Sequence
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.exceptions import EntityExistsError, NoEngineError
@property
def encoded_name(self) -> str:
    """
        Returns the encoded name of the entity.
        """
    return f'"{self.name}"' if '-' in self.name or '_' in self.name else self.name