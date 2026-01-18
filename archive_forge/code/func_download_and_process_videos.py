import csv
import os
import time
import urllib
from functools import partial
from multiprocessing import Pool
from os import path
from typing import Any, Callable, Dict, Optional, Tuple
from torch import Tensor
from .folder import find_classes, make_dataset
from .utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from .video_utils import VideoClips
from .vision import VisionDataset
def download_and_process_videos(self) -> None:
    """Downloads all the videos to the _root_ folder in the expected format."""
    tic = time.time()
    self._download_videos()
    toc = time.time()
    print('Elapsed time for downloading in mins ', (toc - tic) / 60)
    self._make_ds_structure()
    toc2 = time.time()
    print('Elapsed time for processing in mins ', (toc2 - toc) / 60)
    print('Elapsed time overall in mins ', (toc2 - tic) / 60)