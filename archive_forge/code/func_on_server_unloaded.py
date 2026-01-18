from __future__ import annotations
import logging # isort:skip
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any
from ...document import Document
from ..application import ServerContext, SessionContext
def on_server_unloaded(self, server_context: ServerContext) -> None:
    """ Execute code when the server cleanly exits. (Before stopping the
        server's ``IOLoop``.)

        Subclasses may implement this method to provide for any one-time
        tear down that is necessary before the server exits.

        Args:
            server_context (ServerContext) :

        .. warning::
            In practice this code may not run, since servers are often killed
            by a signal.

        """
    pass