from gitdb.db.base import (
from gitdb.util import LazyMixin
from gitdb.exc import (
from gitdb.pack import PackEntity
from functools import reduce
import os
import glob
def sha_iter(self):
    for entity in self.entities():
        index = entity.index()
        sha_by_index = index.sha
        for index in range(index.size()):
            yield sha_by_index(index)