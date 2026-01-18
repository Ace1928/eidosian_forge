import typing
import urllib.parse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import List, Optional, Union
from requests import get
from pennylane.data.base import Dataset
from pennylane.data.base.hdf5 import open_hdf5_s3
from .foldermap import DataPath, FolderMapView, ParamArg
from .params import DEFAULT, FULL, format_params
def remove_paths(foldermap):
    """Copies the foldermap, converting the bottom-level mapping of parameters
        to Paths to a list of the parameters."""
    value = next(iter(foldermap.values()))
    if not isinstance(value, typing.Mapping):
        return sorted(foldermap.keys())
    return {param: remove_paths(foldermap[param]) for param in foldermap.keys()}