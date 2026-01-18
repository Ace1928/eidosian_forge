from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
def show_doc(obj: Model, state: State, notebook_handle: CommsHandle | None=None) -> CommsHandle | None:
    """

    """
    if obj not in state.document.roots:
        state.document.add_root(obj)
    from ..embed.notebook import notebook_content
    comms_target = make_id() if notebook_handle else None
    script, div, cell_doc = notebook_content(obj, comms_target)
    publish_display_data({HTML_MIME_TYPE: div})
    publish_display_data({JS_MIME_TYPE: script, EXEC_MIME_TYPE: ''}, metadata={EXEC_MIME_TYPE: {'id': obj.id}})
    if comms_target:
        handle = CommsHandle(get_comms(comms_target), cell_doc)
        state.document.callbacks.on_change_dispatch_to(handle)
        state.last_comms_handle = handle
        return handle
    return None