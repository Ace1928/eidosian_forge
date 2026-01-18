from __future__ import annotations
import io
import json
from email.parser import Parser
from importlib.resources import files
from typing import TYPE_CHECKING, Any
import js  # type: ignore[import-not-found]
from pyodide.ffi import (  # type: ignore[import-not-found]
from .request import EmscriptenRequest
from .response import EmscriptenResponse
def send_streaming_request(request: EmscriptenRequest) -> EmscriptenResponse | None:
    if _fetcher and streaming_ready():
        return _fetcher.send(request)
    else:
        _show_streaming_warning()
        return None