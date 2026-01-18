from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ...core.serialization import Serializer
from ...document.callbacks import invoke_with_curdoc
from ...document.json import PatchJson
from ..message import Message
def apply_to_document(self, doc: Document, setter: Setter | None=None) -> None:
    """

        """
    invoke_with_curdoc(doc, lambda: doc.apply_json_patch(self.payload, setter=setter))