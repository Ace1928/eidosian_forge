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
def disable_insecure_serializers(allowed=NOTSET):
    """Disable untrusted serializers.

    Will disable all serializers except ``json``
    or you can specify a list of deserializers to allow.

    Note:
    ----
        Producers will still be able to serialize data
        in these formats, but consumers will not accept
        incoming data using the untrusted content types.
    """
    allowed = ['json'] if allowed is NOTSET else allowed
    for name in registry._decoders:
        registry.disable(name)
    if allowed is not None:
        for name in allowed:
            registry.enable(name)