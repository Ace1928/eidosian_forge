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
def create_builder_configs_from_metadata_configs(module_path: str, metadata_configs: MetadataConfigs, supports_metadata: bool, base_path: Optional[str]=None, default_builder_kwargs: Dict[str, Any]=None, download_config: Optional[DownloadConfig]=None) -> Tuple[List[BuilderConfig], str]:
    builder_cls = import_main_class(module_path)
    builder_config_cls = builder_cls.BUILDER_CONFIG_CLASS
    default_config_name = metadata_configs.get_default_config_name()
    builder_configs = []
    default_builder_kwargs = {} if default_builder_kwargs is None else default_builder_kwargs
    base_path = base_path if base_path is not None else ''
    for config_name, config_params in metadata_configs.items():
        config_data_files = config_params.get('data_files')
        config_data_dir = config_params.get('data_dir')
        config_base_path = xjoin(base_path, config_data_dir) if config_data_dir else base_path
        try:
            config_patterns = sanitize_patterns(config_data_files) if config_data_files is not None else get_data_patterns(config_base_path)
            config_data_files_dict = DataFilesPatternsDict.from_patterns(config_patterns, allowed_extensions=ALL_ALLOWED_EXTENSIONS)
        except EmptyDatasetError as e:
            raise EmptyDatasetError(f"Dataset at '{base_path}' doesn't contain data files matching the patterns for config '{config_name}', check `data_files` and `data_fir` parameters in the `configs` YAML field in README.md. ") from e
        if config_data_files is None and supports_metadata and (config_patterns != DEFAULT_PATTERNS_ALL):
            try:
                config_metadata_patterns = get_metadata_patterns(base_path, download_config=download_config)
            except FileNotFoundError:
                config_metadata_patterns = None
            if config_metadata_patterns is not None:
                config_metadata_data_files_list = DataFilesPatternsList.from_patterns(config_metadata_patterns)
                config_data_files_dict = DataFilesPatternsDict({split: data_files_list + config_metadata_data_files_list for split, data_files_list in config_data_files_dict.items()})
        ignored_params = [param for param in config_params if not hasattr(builder_config_cls, param) and param != 'default']
        if ignored_params:
            logger.warning(f'Some datasets params were ignored: {ignored_params}. Make sure to use only valid params for the dataset builder and to have a up-to-date version of the `datasets` library.')
        builder_configs.append(builder_config_cls(name=config_name, data_files=config_data_files_dict, data_dir=config_data_dir, **{param: value for param, value in {**default_builder_kwargs, **config_params}.items() if hasattr(builder_config_cls, param) and param not in ('default', 'data_files', 'data_dir')}))
    return (builder_configs, default_config_name)