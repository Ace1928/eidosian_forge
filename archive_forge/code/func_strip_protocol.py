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
def strip_protocol(urlpath):
    """Return only path part of full URL, according to appropriate backend"""
    protocol, _ = split_protocol(urlpath)
    cls = get_filesystem_class(protocol)
    return cls._strip_protocol(urlpath)