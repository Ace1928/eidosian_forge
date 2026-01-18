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
def github_path_to_dummy_data(self):
    if self._bucket_url is None:
        self._bucket_url = hf_github_url(self.dataset_name, self.dummy_zip_file.replace(os.sep, '/'))
    return self._bucket_url