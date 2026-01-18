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
def sanitize_patterns(patterns: Union[Dict, List, str]) -> Dict[str, Union[List[str], 'DataFilesList']]:
    """
    Take the data_files patterns from the user, and format them into a dictionary.
    Each key is the name of the split, and each value is a list of data files patterns (paths or urls).
    The default split is "train".

    Returns:
        patterns: dictionary of split_name -> list of patterns
    """
    if isinstance(patterns, dict):
        return {str(key): value if isinstance(value, list) else [value] for key, value in patterns.items()}
    elif isinstance(patterns, str):
        return {SANITIZED_DEFAULT_SPLIT: [patterns]}
    elif isinstance(patterns, list):
        if any((isinstance(pattern, dict) for pattern in patterns)):
            for pattern in patterns:
                if not (isinstance(pattern, dict) and len(pattern) == 2 and ('split' in pattern) and isinstance(pattern.get('path'), (str, list))):
                    raise ValueError(f"Expected each split to have a 'path' key which can be a string or a list of strings, but got {pattern}")
            splits = [pattern['split'] for pattern in patterns]
            if len(set(splits)) != len(splits):
                raise ValueError(f'Some splits are duplicated in data_files: {splits}')
            return {str(pattern['split']): pattern['path'] if isinstance(pattern['path'], list) else [pattern['path']] for pattern in patterns}
        else:
            return {SANITIZED_DEFAULT_SPLIT: patterns}
    else:
        return sanitize_patterns(list(patterns))