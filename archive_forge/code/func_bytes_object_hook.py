from __future__ import annotations
import logging # isort:skip
import base64
import calendar
import codecs
import datetime as dt
import hashlib
import hmac
import json
import time
import zlib
from typing import TYPE_CHECKING, Any
from ..core.types import ID
from ..settings import settings
from .warnings import warn
def bytes_object_hook(self, obj: dict[Any, Any]) -> Any:
    if set(obj.keys()) == {'bytes'}:
        return _base64_decode(obj['bytes'])
    return obj