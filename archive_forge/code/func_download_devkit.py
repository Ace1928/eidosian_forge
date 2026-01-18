import os
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from .folder import default_loader
from .utils import check_integrity, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def download_devkit(self) -> None:
    file, md5 = self._DEVKIT_META[self.variant]
    download_and_extract_archive(urljoin(self._BASE_URL, file), self.root, md5=md5)