import bz2
import contextlib
import gzip
import hashlib
import itertools
import lzma
import os
import os.path
import pathlib
import re
import sys
import tarfile
import urllib
import urllib.error
import urllib.request
import warnings
import zipfile
from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Optional, Tuple, TypeVar
from urllib.parse import urlparse
import numpy as np
import requests
import torch
from torch.utils.model_zoo import tqdm
from .._internally_replaced_utils import _download_file_from_remote_location, _is_remote_location_available
def extract_archive(from_path: str, to_path: Optional[str]=None, remove_finished: bool=False) -> str:
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name. If the file is compressed
    but not an archive the call is dispatched to :func:`decompress`.

    Args:
        from_path (str): Path to the file to be extracted.
        to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the directory the file was extracted to.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)
    suffix, archive_type, compression = _detect_file_type(from_path)
    if not archive_type:
        return _decompress(from_path, os.path.join(to_path, os.path.basename(from_path).replace(suffix, '')), remove_finished=remove_finished)
    extractor = _ARCHIVE_EXTRACTORS[archive_type]
    extractor(from_path, to_path, compression)
    if remove_finished:
        os.remove(from_path)
    return to_path