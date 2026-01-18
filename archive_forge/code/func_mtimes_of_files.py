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
def mtimes_of_files(dirnames: List[str], suffix: str) -> Iterator[float]:
    for dirname in dirnames:
        for root, _dirs, files in os.walk(dirname):
            for sfile in files:
                if sfile.endswith(suffix):
                    try:
                        yield path.getmtime(path.join(root, sfile))
                    except OSError:
                        pass