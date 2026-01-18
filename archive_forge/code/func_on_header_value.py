from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
def on_header_value(self, data: bytes, start: int, end: int) -> None:
    self._current_partial_header_value += data[start:end]