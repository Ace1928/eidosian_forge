from breezy.lazy_import import lazy_import
from ... import config, merge
import fnmatch
import subprocess
import tempfile
from breezy import (
def merge_matching(self, params):
    return self.merge_text(params)