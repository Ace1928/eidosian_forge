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
def register_pickle():
    """Register pickle serializer.

    The fastest serialization method, but restricts
    you to python clients.
    """

    def pickle_dumps(obj, dumper=pickle.dumps):
        return dumper(obj, protocol=pickle_protocol)
    registry.register('pickle', pickle_dumps, unpickle, content_type='application/x-python-serialize', content_encoding='binary')