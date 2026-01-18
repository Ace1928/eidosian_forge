import io
import os
import re
import tarfile
import tempfile
from .fnmatch import fnmatch
from ..constants import IS_WINDOWS_PLATFORM
def normalize_slashes(p):
    if IS_WINDOWS_PLATFORM:
        return '/'.join(split_path(p))
    return p