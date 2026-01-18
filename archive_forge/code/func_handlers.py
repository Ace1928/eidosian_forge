from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from typing import (
from ..core.types import ID
from ..document import Document
from ..settings import settings
@property
def handlers(self) -> tuple[Handler, ...]:
    """ The ordered list of handlers this Application is configured with.

        """
    return tuple(self._handlers)