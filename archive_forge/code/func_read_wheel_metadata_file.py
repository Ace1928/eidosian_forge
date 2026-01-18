import logging
from email.message import Message
from email.parser import Parser
from typing import Tuple
from zipfile import BadZipFile, ZipFile
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import UnsupportedWheel
def read_wheel_metadata_file(source: ZipFile, path: str) -> bytes:
    try:
        return source.read(path)
    except (BadZipFile, KeyError, RuntimeError) as e:
        raise UnsupportedWheel(f'could not read {path!r} file: {e!r}')