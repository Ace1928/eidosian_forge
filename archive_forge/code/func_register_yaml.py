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
def register_yaml():
    """Register a encoder/decoder for YAML serialization.

    It is slower than JSON, but allows for more data types
    to be serialized. Useful if you need to send data such as dates

    """
    try:
        import yaml
        registry.register('yaml', yaml.safe_dump, yaml.safe_load, content_type='application/x-yaml', content_encoding='utf-8')
    except ImportError:

        def not_available(*args, **kwargs):
            """Raise SerializerNotInstalled.

            Used in case a client receives a yaml message, but yaml
            isn't installed.
            """
            raise SerializerNotInstalled('No decoder installed for YAML. Install the PyYAML library')
        registry.register('yaml', None, not_available, 'application/x-yaml')