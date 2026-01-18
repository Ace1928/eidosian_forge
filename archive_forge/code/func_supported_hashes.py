import functools
import itertools
import logging
import os
import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from typing import (
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
from pip._internal.utils.models import KeyBasedCompareMixin
from pip._internal.utils.urls import path_to_url, url_to_path
def supported_hashes(hashes: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if hashes is None:
        return None
    hashes = {n: v for n, v in hashes.items() if n in _SUPPORTED_HASHES}
    if not hashes:
        return None
    return hashes