import contextlib
import copy
import fnmatch
import json
import math
import posixpath
import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import fsspec
import numpy as np
from huggingface_hub import (
from . import config
from .arrow_dataset import PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED, Dataset
from .features import Features
from .features.features import FeatureType
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import Table
from .tasks import TaskTemplate
from .utils import logging
from .utils.deprecation_utils import deprecated
from .utils.doc_utils import is_documented_by
from .utils.hub import list_files_info
from .utils.metadata import MetadataConfigs
from .utils.py_utils import asdict, glob_pattern_to_regex, string_to_dict
from .utils.typing import PathLike
@staticmethod
def from_csv(path_or_paths: Dict[str, PathLike], features: Optional[Features]=None, cache_dir: str=None, keep_in_memory: bool=False, **kwargs) -> 'DatasetDict':
    """Create [`DatasetDict`] from CSV file(s).

        Args:
            path_or_paths (`dict` of path-like):
                Path(s) of the CSV file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (str, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`pandas.read_csv`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_csv({'train': 'path/to/dataset.csv'})
        ```
        """
    from .io.csv import CsvDatasetReader
    return CsvDatasetReader(path_or_paths, features=features, cache_dir=cache_dir, keep_in_memory=keep_in_memory, **kwargs).read()