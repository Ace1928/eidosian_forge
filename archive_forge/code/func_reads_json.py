from __future__ import annotations
import re
import warnings
from traitlets.log import get_logger
from nbformat import v3 as _v_latest
from nbformat.v3 import (
from . import versions
from .converter import convert
from .reader import reads as reader_reads
from .validator import ValidationError, validate
def reads_json(nbjson, **kwargs):
    """DEPRECATED, use reads"""
    warnings.warn('reads_json is deprecated since nbformat 3.0, use reads', DeprecationWarning, stacklevel=2)
    return reads(nbjson)