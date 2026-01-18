from __future__ import annotations
import codecs
import os
import pickle
import sys
from collections import namedtuple
from contextlib import contextmanager
from io import BytesIO
from .exceptions import (ContentDisallowed, DecodeError, EncodeError,
from .utils.compat import entrypoints
from .utils.encoding import bytes_to_str, str_to_bytes
def register_json():
    """Register a encoder/decoder for JSON serialization."""
    from kombu.utils import json as _json
    registry.register('json', _json.dumps, _json.loads, content_type='application/json', content_encoding='utf-8')