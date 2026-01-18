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
def get_metadata_patterns(base_path: str, download_config: Optional[DownloadConfig]=None) -> List[str]:
    """
    Get the supported metadata patterns from a local directory.
    """
    resolver = partial(resolve_pattern, base_path=base_path, download_config=download_config)
    try:
        return _get_metadata_files_patterns(resolver)
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory at {base_path} doesn't contain any metadata file") from None