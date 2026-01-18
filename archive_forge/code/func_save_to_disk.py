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
def save_to_disk(self, dataset_dict_path: PathLike, fs='deprecated', max_shard_size: Optional[Union[str, int]]=None, num_shards: Optional[Dict[str, int]]=None, num_proc: Optional[int]=None, storage_options: Optional[dict]=None):
    """
        Saves a dataset dict to a filesystem using `fsspec.spec.AbstractFileSystem`.

        For [`Image`] and [`Audio`] data:

        All the Image() and Audio() data are stored in the arrow files.
        If you want to store paths or urls, please use the Value("string") type.

        Args:
            dataset_dict_path (`str`):
                Path (e.g. `dataset/train`) or remote URI
                (e.g. `s3://my-bucket/dataset/train`) of the dataset dict directory where the dataset dict will be
                saved to.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem where the dataset will be saved to.

                <Deprecated version="2.8.0">

                `fs` was deprecated in version 2.8.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`

                </Deprecated>

            max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
                The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
                (like `"50MB"`).
            num_shards (`Dict[str, int]`, *optional*):
                Number of shards to write. By default the number of shards depends on `max_shard_size` and `num_proc`.
                You need to provide the number of shards for each dataset in the dataset dictionary.
                Use a dictionary to define a different num_shards for each split.

                <Added version="2.8.0"/>
            num_proc (`int`, *optional*, default `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.8.0"/>
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.8.0"/>

        Example:

        ```python
        >>> dataset_dict.save_to_disk("path/to/dataset/directory")
        >>> dataset_dict.save_to_disk("path/to/dataset/directory", max_shard_size="1GB")
        >>> dataset_dict.save_to_disk("path/to/dataset/directory", num_shards={"train": 1024, "test": 8})
        ```
        """
    if fs != 'deprecated':
        warnings.warn("'fs' was deprecated in favor of 'storage_options' in version 2.8.0 and will be removed in 3.0.0.\nYou can remove this warning by passing 'storage_options=fs.storage_options' instead.", FutureWarning)
        storage_options = fs.storage_options
    fs: fsspec.AbstractFileSystem
    fs, _, _ = fsspec.get_fs_token_paths(dataset_dict_path, storage_options=storage_options)
    if num_shards is None:
        num_shards = {k: None for k in self}
    elif not isinstance(num_shards, dict):
        raise ValueError("Please provide one `num_shards` per dataset in the dataset dictionary, e.g. {{'train': 128, 'test': 4}}")
    fs.makedirs(dataset_dict_path, exist_ok=True)
    with fs.open(posixpath.join(dataset_dict_path, config.DATASETDICT_JSON_FILENAME), 'w', encoding='utf-8') as f:
        json.dump({'splits': list(self)}, f)
    for k, dataset in self.items():
        dataset.save_to_disk(posixpath.join(dataset_dict_path, k), num_shards=num_shards.get(k), max_shard_size=max_shard_size, num_proc=num_proc, storage_options=storage_options)