import os
import re
from functools import partial
from glob import has_magic
from pathlib import Path, PurePath
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import huggingface_hub
from fsspec import get_fs_token_paths
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import HfFileSystem
from packaging import version
from tqdm.contrib.concurrent import thread_map
from . import config
from .download import DownloadConfig
from .download.streaming_download_manager import _prepare_path_and_storage_options, xbasename, xjoin
from .naming import _split_re
from .splits import Split
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import is_local_path, is_relative_path
from .utils.py_utils import glob_pattern_to_regex, string_to_dict
@classmethod
def from_hf_repo(cls, patterns: Dict[str, Union[List[str], DataFilesList]], dataset_info: huggingface_hub.hf_api.DatasetInfo, base_path: Optional[str]=None, allowed_extensions: Optional[List[str]]=None, download_config: Optional[DownloadConfig]=None) -> 'DataFilesDict':
    out = cls()
    for key, patterns_for_key in patterns.items():
        out[key] = DataFilesList.from_hf_repo(patterns_for_key, dataset_info=dataset_info, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config) if not isinstance(patterns_for_key, DataFilesList) else patterns_for_key
    return out