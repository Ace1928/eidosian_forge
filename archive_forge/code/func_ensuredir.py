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
def ensuredir(path: str) -> None:
    """Ensure that a path exists."""
    os.makedirs(path, exist_ok=True)