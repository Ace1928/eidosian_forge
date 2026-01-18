import io
import os
import re
import tarfile
import tempfile
from .fnmatch import fnmatch
from ..constants import IS_WINDOWS_PLATFORM
def match_tag(tag: str) -> bool:
    return bool(_TAG.match(tag))