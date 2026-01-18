import os
import re
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional, Union
from zipfile import ZipFile
from ..utils.file_utils import cached_path, hf_github_url
from ..utils.logging import get_logger
from ..utils.version import Version
def download_dummy_data(self):
    path_to_dummy_data_dir = self.local_path_to_dummy_data if self.use_local_dummy_data is True else self.github_path_to_dummy_data
    local_path = cached_path(path_to_dummy_data_dir, cache_dir=self.cache_dir, extract_compressed_file=True, force_extract=True)
    return os.path.join(local_path, self.dummy_file_name)