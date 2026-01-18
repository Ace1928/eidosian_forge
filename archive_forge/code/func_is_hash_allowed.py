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
def is_hash_allowed(self, hashes: Optional[Hashes]) -> bool:
    """
        Return True if the link has a hash and it is allowed by `hashes`.
        """
    if hashes is None:
        return False
    return any((hashes.is_hash_allowed(k, v) for k, v in self._hashes.items()))