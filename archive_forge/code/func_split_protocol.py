from __future__ import annotations
import io
import logging
import os
import re
from glob import has_magic
from pathlib import Path
from .caching import (  # noqa: F401
from .compression import compr
from .registry import filesystem, get_filesystem_class
from .utils import (
def split_protocol(urlpath):
    """Return protocol, path pair"""
    urlpath = stringify_path(urlpath)
    if '://' in urlpath:
        protocol, path = urlpath.split('://', 1)
        if len(protocol) > 1:
            return (protocol, path)
    if urlpath.startswith('data:'):
        return urlpath.split(':', 1)
    return (None, urlpath)