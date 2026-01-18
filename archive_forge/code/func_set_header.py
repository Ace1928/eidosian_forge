from __future__ import annotations
from dataclasses import dataclass, field
from ..._base_connection import _TYPE_BODY
def set_header(self, name: str, value: str) -> None:
    self.headers[name.capitalize()] = value