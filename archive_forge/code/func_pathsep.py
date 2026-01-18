import os
import posixpath
import sys
import urllib.parse
import warnings
from os.path import join as pjoin
import pyarrow as pa
from pyarrow.util import doc, _stringify_path, _is_path_like, _DEPR_MSG
@property
def pathsep(self):
    return os.path.sep