from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
def on_field_data(self, data: bytes, start: int, end: int) -> None:
    message = (FormMessage.FIELD_DATA, data[start:end])
    self.messages.append(message)