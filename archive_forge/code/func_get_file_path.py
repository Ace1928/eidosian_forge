import csv
import glob
import os
import warnings
from contextlib import ExitStack
from typing import List, Tuple
import fsspec
import pandas
import pandas._libs.lib as lib
from pandas.io.common import is_fsspec_url, is_url, stringify_path
from modin.config import NPartitions
from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.csv_dispatcher import CSVDispatcher
def get_file_path(fs_handle) -> List[str]:
    if '*' in file_path:
        file_paths = fs_handle.glob(file_path)
    else:
        file_paths = [f for f in fs_handle.find(file_path) if not f.endswith('/')]
    if len(file_paths) == 0 and (not fs_handle.exists(file_path)):
        raise FileNotFoundError(f"Path <{file_path}> isn't available.")
    fs_addresses = [fs_handle.unstrip_protocol(path) for path in file_paths]
    return fs_addresses