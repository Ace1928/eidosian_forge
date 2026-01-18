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
def prepare_accept_content(content_types, name_to_type=None):
    """Replace aliases of content_types with full names from registry.

    Raises
    ------
        SerializerNotInstalled: If the serialization method
            requested is not available.
    """
    name_to_type = registry.name_to_type if not name_to_type else name_to_type
    if content_types is not None:
        try:
            return {n if '/' in n else name_to_type[n] for n in content_types}
        except KeyError as e:
            raise SerializerNotInstalled(f'No encoder/decoder installed for {e.args[0]}')
    return content_types