from __future__ import annotations
import base64
from abc import ABC, abstractmethod
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.stores import BaseStore, ByteStore
from langchain_community.utilities.astradb import (
def encode_value(self, value: Optional[bytes]) -> Any:
    if value is None:
        return None
    return base64.b64encode(value).decode('ascii')