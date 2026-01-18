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
def resolve_pattern(pattern: str, base_path: str, allowed_extensions: Optional[List[str]]=None, download_config: Optional[DownloadConfig]=None) -> List[str]:
    """
    Resolve the paths and URLs of the data files from the pattern passed by the user.

    You can use patterns to resolve multiple local files. Here are a few examples:
    - *.csv to match all the CSV files at the first level
    - **.csv to match all the CSV files at any level
    - data/* to match all the files inside "data"
    - data/** to match all the files inside "data" and its subdirectories

    The patterns are resolved using the fsspec glob. In fsspec>=2023.12.0 this is equivalent to
    Python's glob.glob, Path.glob, Path.match and fnmatch where ** is unsupported with a prefix/suffix
    other than a forward slash /.

    More generally:
    - '*' matches any character except a forward-slash (to match just the file or directory name)
    - '**' matches any character including a forward-slash /

    Hidden files and directories (i.e. whose names start with a dot) are ignored, unless they are explicitly requested.
    The same applies to special directories that start with a double underscore like "__pycache__".
    You can still include one if the pattern explicilty mentions it:
    - to include a hidden file: "*/.hidden.txt" or "*/.*"
    - to include a hidden directory: ".hidden/*" or ".*/*"
    - to include a special directory: "__special__/*" or "__*/*"

    Example::

        >>> from datasets.data_files import resolve_pattern
        >>> base_path = "."
        >>> resolve_pattern("docs/**/*.py", base_path)
        [/Users/mariosasko/Desktop/projects/datasets/docs/source/_config.py']

    Args:
        pattern (str): Unix pattern or paths or URLs of the data files to resolve.
            The paths can be absolute or relative to base_path.
            Remote filesystems using fsspec are supported, e.g. with the hf:// protocol.
        base_path (str): Base path to use when resolving relative paths.
        allowed_extensions (Optional[list], optional): White-list of file extensions to use. Defaults to None (all extensions).
            For example: allowed_extensions=[".csv", ".json", ".txt", ".parquet"]
    Returns:
        List[str]: List of paths or URLs to the local or remote files that match the patterns.
    """
    if is_relative_path(pattern):
        pattern = xjoin(base_path, pattern)
    elif is_local_path(pattern):
        base_path = os.path.splitdrive(pattern)[0] + os.sep
    else:
        base_path = ''
    pattern, storage_options = _prepare_path_and_storage_options(pattern, download_config=download_config)
    fs, _, _ = get_fs_token_paths(pattern, storage_options=storage_options)
    fs_base_path = base_path.split('::')[0].split('://')[-1] or fs.root_marker
    fs_pattern = pattern.split('::')[0].split('://')[-1]
    files_to_ignore = set(FILES_TO_IGNORE) - {xbasename(pattern)}
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    protocol_prefix = protocol + '://' if protocol != 'file' else ''
    glob_kwargs = {}
    if protocol == 'hf' and config.HF_HUB_VERSION >= version.parse('0.20.0'):
        glob_kwargs['expand_info'] = False
    matched_paths = [filepath if filepath.startswith(protocol_prefix) else protocol_prefix + filepath for filepath, info in fs.glob(pattern, detail=True, **glob_kwargs).items() if info['type'] == 'file' and xbasename(filepath) not in files_to_ignore and (not _is_inside_unrequested_special_dir(os.path.relpath(filepath, fs_base_path), os.path.relpath(fs_pattern, fs_base_path))) and (not _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(os.path.relpath(filepath, fs_base_path), os.path.relpath(fs_pattern, fs_base_path)))]
    if allowed_extensions is not None:
        out = [filepath for filepath in matched_paths if any(('.' + suffix in allowed_extensions for suffix in xbasename(filepath).split('.')[1:]))]
        if len(out) < len(matched_paths):
            invalid_matched_files = list(set(matched_paths) - set(out))
            logger.info(f"Some files matched the pattern '{pattern}' but don't have valid data file extensions: {invalid_matched_files}")
    else:
        out = matched_paths
    if not out:
        error_msg = f"Unable to find '{pattern}'"
        if allowed_extensions is not None:
            error_msg += f' with any supported extension {list(allowed_extensions)}'
        raise FileNotFoundError(error_msg)
    return out