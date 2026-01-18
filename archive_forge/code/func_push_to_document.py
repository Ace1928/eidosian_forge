from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, TypedDict
from ...core.types import ID
from ..exceptions import ProtocolError
from ..message import Message
def push_to_document(self, doc: Document) -> None:
    if 'doc' not in self.content:
        raise ProtocolError('No doc in PULL-DOC-REPLY')
    doc.replace_with_json(self.content['doc'])