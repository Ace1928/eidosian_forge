from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
def on_part_data(self, data: bytes, start: int, end: int) -> None:
    message_bytes = data[start:end]
    if self._current_part.file is None:
        self._current_part.data += message_bytes
    else:
        self._file_parts_to_write.append((self._current_part, message_bytes))