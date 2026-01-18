import os
import re
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional, Union
from zipfile import ZipFile
from ..utils.file_utils import cached_path, hf_github_url
from ..utils.logging import get_logger
from ..utils.version import Version
@property
def dummy_file(self):
    if self._dummy_file is None:
        self._dummy_file = self.download_dummy_data()
    return self._dummy_file