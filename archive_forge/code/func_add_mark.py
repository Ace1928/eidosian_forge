import atexit
import os
import shutil
import tempfile
import weakref
from fastimport.reftracker import RefTracker
from ... import lru_cache, trace
from . import branch_mapper
from .helpers import single_plural
def add_mark(self, mark, commit_id):
    if mark.startswith(b':'):
        raise ValueError(mark)
    is_new = mark in self.marks
    self.marks[mark] = commit_id
    return is_new