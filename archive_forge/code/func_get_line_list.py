import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
def get_line_list(self, version_ids):
    return [self.cache_version(v) for v in version_ids]