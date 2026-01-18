import csv
import os
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive
from .vision import VisionDataset
def download(self) -> None:
    """Download the KITTI data if it doesn't exist already."""
    if self._check_exists():
        return
    os.makedirs(self._raw_folder, exist_ok=True)
    for fname in self.resources:
        download_and_extract_archive(url=f'{self.data_url}{fname}', download_root=self._raw_folder, filename=fname)