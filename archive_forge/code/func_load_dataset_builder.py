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
def load_dataset_builder(path: str, name: Optional[str]=None, data_dir: Optional[str]=None, data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]=None, cache_dir: Optional[str]=None, features: Optional[Features]=None, download_config: Optional[DownloadConfig]=None, download_mode: Optional[Union[DownloadMode, str]]=None, revision: Optional[Union[str, Version]]=None, token: Optional[Union[bool, str]]=None, use_auth_token='deprecated', storage_options: Optional[Dict]=None, trust_remote_code: Optional[bool]=None, _require_default_config_name=True, **config_kwargs) -> DatasetBuilder:
    """Load a dataset builder from the Hugging Face Hub, or a local dataset. A dataset builder can be used to inspect general information that is required to build a dataset (cache directory, config, dataset info, etc.)
    without downloading the dataset itself.

    You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with [`huggingface_hub.list_datasets`].

    A dataset is a directory that contains:

    - some data files in generic formats (JSON, CSV, Parquet, text, etc.)
    - and optionally a dataset script, if it requires some code to read the data files. This is used to load any kind of formats or structures.

    Note that dataset scripts can also download and read data files from anywhere - in case your data files already exist online.

    Args:

        path (`str`):
            Path or name of the dataset.
            Depending on `path`, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.

            For local datasets:

            - if `path` is a local directory (containing data files only)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. `'./path/to/directory/with/my/csv/data'`.
            - if `path` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script
              e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`.

            For datasets on the Hugging Face Hub (list all available datasets with [`huggingface_hub.list_datasets`])

            - if `path` is a dataset repository on the HF hub (containing data files only)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing your data files.
            - if `path` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script in the dataset repository
              e.g. `glue`, `squad`, `'username/dataset_name'`, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`,
            the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.
        features ([`Features`], *optional*):
            Set the features type to use for this dataset.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset script to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        use_auth_token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.

            <Deprecated version="2.14.0">

            `use_auth_token` was deprecated in favor of `token` in version 2.14.0 and will be removed in 3.0.0.

            </Deprecated>
        storage_options (`dict`, *optional*, defaults to `None`):
            **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.

            <Added version="2.11.0"/>
        trust_remote_code (`bool`, defaults to `True`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Tip warning={true}>

            `trust_remote_code` will default to False in the next major release.

            </Tip>

            <Added version="2.16.0"/>
        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the [`BuilderConfig`]
            and used in the [`DatasetBuilder`].

    Returns:
        [`DatasetBuilder`]

    Example:

    ```py
    >>> from datasets import load_dataset_builder
    >>> ds_builder = load_dataset_builder('rotten_tomatoes')
    >>> ds_builder.info.features
    {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
     'text': Value(dtype='string', id=None)}
    ```
    """
    if use_auth_token != 'deprecated':
        warnings.warn("'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\nYou can remove this warning by passing 'token=<use_auth_token>' instead.", FutureWarning)
        token = use_auth_token
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    if token is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.token = token
    if storage_options is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.storage_options.update(storage_options)
    dataset_module = dataset_module_factory(path, revision=revision, download_config=download_config, download_mode=download_mode, data_dir=data_dir, data_files=data_files, cache_dir=cache_dir, trust_remote_code=trust_remote_code, _require_default_config_name=_require_default_config_name, _require_custom_configs=bool(config_kwargs))
    builder_kwargs = dataset_module.builder_kwargs
    data_dir = builder_kwargs.pop('data_dir', data_dir)
    data_files = builder_kwargs.pop('data_files', data_files)
    config_name = builder_kwargs.pop('config_name', name or dataset_module.builder_configs_parameters.default_config_name)
    dataset_name = builder_kwargs.pop('dataset_name', None)
    info = dataset_module.dataset_infos.get(config_name) if dataset_module.dataset_infos else None
    if path in _PACKAGED_DATASETS_MODULES and data_files is None and (dataset_module.builder_configs_parameters.builder_configs[0].data_files is None):
        error_msg = f'Please specify the data files or data directory to load for the {path} dataset builder.'
        example_extensions = [extension for extension in _EXTENSION_TO_MODULE if _EXTENSION_TO_MODULE[extension] == path]
        if example_extensions:
            error_msg += f'\nFor example `data_files={{"train": "path/to/data/train/*.{example_extensions[0]}"}}`'
        raise ValueError(error_msg)
    builder_cls = get_dataset_builder_class(dataset_module, dataset_name=dataset_name)
    builder_instance: DatasetBuilder = builder_cls(cache_dir=cache_dir, dataset_name=dataset_name, config_name=config_name, data_dir=data_dir, data_files=data_files, hash=dataset_module.hash, info=info, features=features, token=token, storage_options=storage_options, **builder_kwargs, **config_kwargs)
    builder_instance._use_legacy_cache_dir_if_possible(dataset_module)
    return builder_instance