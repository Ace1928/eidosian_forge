import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def deprecate_no_replacement(name: str, removed_in: str) -> None:
    """Raise an exception that a feature will be removed without replacement."""
    deprecate(DEPR_MSG_NO_REPLACEMENT.format(name, removed_in), 4)