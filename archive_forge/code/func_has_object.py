from gitdb.db.base import (
from gitdb.util import LazyMixin
from gitdb.exc import (
from gitdb.pack import PackEntity
from functools import reduce
import os
import glob
def has_object(self, sha):
    try:
        self._pack_info(sha)
        return True
    except BadObject:
        return False