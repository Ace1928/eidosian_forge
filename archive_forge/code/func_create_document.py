from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from typing import (
from ..core.types import ID
from ..document import Document
from ..settings import settings
def create_document(self) -> Document:
    """ Creates and initializes a document using the Application's handlers.

        """
    doc = Document()
    self.initialize_document(doc)
    return doc