from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from typing import (
from ..core.types import ID
from ..document import Document
from ..settings import settings
def initialize_document(self, doc: Document) -> None:
    """ Fills in a new document using the Application's handlers.

        """
    for h in self._handlers:
        h.modify_document(doc)
        if h.failed:
            log.error('Error running application handler %r: %s %s ', h, h.error, h.error_detail)
    if settings.perform_document_validation():
        doc.validate()