import os
from io import BytesIO
import zipfile
import tempfile
import shutil
import enum
import warnings
from ..core import urlopen, get_remote_file
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
@format_hint.setter
def format_hint(self, format: str) -> None:
    self._format_hint = format
    if self._extension is None:
        self._extension = format