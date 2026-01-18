from __future__ import annotations
import logging # isort:skip
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any
from ...document import Document
from ..application import ServerContext, SessionContext
def modify_document(self, doc: Document) -> None:
    """ Modify an application document in a specified manner.

        When a Bokeh server session is initiated, the Bokeh server asks the
        Application for a new Document to service the session. To do this,
        the Application first creates a new empty Document, then it passes
        this Document to the ``modify_document`` method of each of its
        handlers. When all handlers have updated the Document, it is used to
        service the user session.

        *Subclasses must implement this method*

        Args:
            doc (Document) : A Bokeh Document to update in-place

        Returns:
            Document

        """
    raise NotImplementedError('implement modify_document()')