from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Awaitable
def send_patch_document(self, event: DocumentPatchedEvent) -> Awaitable[None]:
    """ Sends a PATCH-DOC message, returning a Future that's completed when it's written out. """
    msg = self.protocol.create('PATCH-DOC', [event])
    return self._socket.send_message(msg)