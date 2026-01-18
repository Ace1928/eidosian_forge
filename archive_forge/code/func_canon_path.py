import contextlib
import filecmp
import os
import re
import shutil
import sys
import unicodedata
from io import StringIO
from os import path
from typing import Any, Generator, Iterator, List, Optional, Type
def canon_path(nativepath: str) -> str:
    """Return path in OS-independent form"""
    return nativepath.replace(path.sep, SEP)