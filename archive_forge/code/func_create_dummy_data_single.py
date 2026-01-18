import os
import re
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional, Union
from zipfile import ZipFile
from ..utils.file_utils import cached_path, hf_github_url
from ..utils.logging import get_logger
from ..utils.version import Version
def create_dummy_data_single(self, path_to_dummy_data, data_url):
    for download_callback in self.download_callbacks:
        download_callback(data_url)
    value = os.path.join(path_to_dummy_data, urllib.parse.quote_plus(data_url.split('/')[-1]))
    if os.path.exists(value) or not self.load_existing_dummy_data:
        return value
    else:
        return path_to_dummy_data