from __future__ import annotations
import contextlib
import logging
import math
import os
import pathlib
import re
import sys
import tempfile
from functools import partial
from hashlib import md5
from importlib.metadata import version
from typing import (
from urllib.parse import urlsplit
def can_be_local(path: str) -> bool:
    """Can the given URL be used with open_local?"""
    from fsspec import get_filesystem_class
    try:
        return getattr(get_filesystem_class(get_protocol(path)), 'local_file', False)
    except (ValueError, ImportError):
        return False