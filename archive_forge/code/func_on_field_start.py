from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
def on_field_start(self) -> None:
    message = (FormMessage.FIELD_START, b'')
    self.messages.append(message)