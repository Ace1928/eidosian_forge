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
@property
def image_mode(self) -> ImageMode:
    return ImageMode(self.value[1])