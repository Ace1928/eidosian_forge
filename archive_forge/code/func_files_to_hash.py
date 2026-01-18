import filecmp
import glob
import importlib
import inspect
import json
import os
import posixpath
import shutil
import signal
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import fsspec
import requests
import yaml
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFileSystem
from . import config
from .arrow_dataset import Dataset
from .builder import BuilderConfig, DatasetBuilder
from .data_files import (
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager, xbasename, xglob, xjoin
from .exceptions import DataFilesNotFoundError, DatasetNotFoundError
from .features import Features
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict
from .iterable_dataset import IterableDataset
from .metric import Metric
from .naming import camelcase_to_snakecase, snakecase_to_camelcase
from .packaged_modules import (
from .splits import Split
from .utils import _datasets_server
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.file_utils import (
from .utils.hub import hf_hub_url
from .utils.info_utils import VerificationMode, is_small_dataset
from .utils.logging import get_logger
from .utils.metadata import MetadataConfigs
from .utils.py_utils import get_imports
from .utils.version import Version
def files_to_hash(file_paths: List[str]) -> str:
    """
    Convert a list of scripts or text files provided in file_paths into a hashed filename in a repeatable way.
    """
    to_use_files: List[Union[Path, str]] = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            to_use_files.extend(list(Path(file_path).rglob('*.[pP][yY]')))
        else:
            to_use_files.append(file_path)
    lines = []
    for file_path in to_use_files:
        with open(file_path, encoding='utf-8') as f:
            lines.extend(f.readlines())
    return _hash_python_lines(lines)