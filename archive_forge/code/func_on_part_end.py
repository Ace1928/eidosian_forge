from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
def on_part_end(self) -> None:
    if self._current_part.file is None:
        self.items.append((self._current_part.field_name, _user_safe_decode(self._current_part.data, self._charset)))
    else:
        self._file_parts_to_finish.append(self._current_part)
        self.items.append((self._current_part.field_name, self._current_part.file))