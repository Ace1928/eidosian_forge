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
def metadata_link(self) -> Optional['Link']:
    """Return a link to the associated core metadata file (if any)."""
    if self.metadata_file_data is None:
        return None
    metadata_url = f'{self.url_without_fragment}.metadata'
    if self.metadata_file_data.hashes is None:
        return Link(metadata_url)
    return Link(metadata_url, hashes=self.metadata_file_data.hashes)