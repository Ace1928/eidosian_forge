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
def dataset_module_factory(path: str, revision: Optional[Union[str, Version]]=None, download_config: Optional[DownloadConfig]=None, download_mode: Optional[Union[DownloadMode, str]]=None, dynamic_modules_path: Optional[str]=None, data_dir: Optional[str]=None, data_files: Optional[Union[Dict, List, str, DataFilesDict]]=None, cache_dir: Optional[str]=None, trust_remote_code: Optional[bool]=None, _require_default_config_name=True, _require_custom_configs=False, **download_kwargs) -> DatasetModule:
    """
    Download/extract/cache a dataset module.

    Dataset codes are cached inside the dynamic modules cache to allow easy import (avoid ugly sys.path tweaks).

    Args:

        path (str): Path or name of the dataset.
            Depending on ``path``, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.

            For local datasets:

            - if ``path`` is a local directory (containing data files only)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. ``'./path/to/directory/with/my/csv/data'``.
            - if ``path`` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory):
              -> load the dataset builder from the dataset script
              e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'``.

            For datasets on the Hugging Face Hub (list all available datasets with ``huggingface_hub.list_datasets()``)

            - if ``path`` is a dataset repository on the HF hub (containing data files only)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. ``'username/dataset_name'``, a dataset repository on the HF hub containing your data files.
            - if ``path`` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script in the dataset repository
              e.g. ``glue``, ``squad``, ``'username/dataset_name'``, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

        revision (:class:`~utils.Version` or :obj:`str`, optional): Version of the dataset script to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        download_config (:class:`DownloadConfig`, optional): Specific download configuration parameters.
        download_mode (:class:`DownloadMode` or :obj:`str`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        dynamic_modules_path (Optional str, defaults to HF_MODULES_CACHE / "datasets_modules", i.e. ~/.cache/huggingface/modules/datasets_modules):
            Optional path to the directory in which the dynamic modules are saved. It must have been initialized with :obj:`init_dynamic_modules`.
            By default, the datasets and metrics are stored inside the `datasets_modules` module.
        data_dir (:obj:`str`, optional): Directory with the data files. Used only if `data_files` is not specified,
            in which case it's equal to pass `os.path.join(data_dir, "**")` as `data_files`.
        data_files (:obj:`Union[Dict, List, str]`, optional): Defining the data_files of the dataset configuration.
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.

            <Added version="2.16.0"/>
        trust_remote_code (`bool`, defaults to `True`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Tip warning={true}>

            `trust_remote_code` will default to False in the next major release.

            </Tip>

            <Added version="2.16.0"/>
        **download_kwargs (additional keyword arguments): optional attributes for DownloadConfig() which will override
            the attributes in download_config if supplied.

    Returns:
        DatasetModule
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    download_config.extract_compressed_file = True
    download_config.force_extract = True
    download_config.force_download = download_mode == DownloadMode.FORCE_REDOWNLOAD
    filename = list(filter(lambda x: x, path.replace(os.sep, '/').split('/')))[-1]
    if not filename.endswith('.py'):
        filename = filename + '.py'
    combined_path = os.path.join(path, filename)
    if path in _PACKAGED_DATASETS_MODULES:
        return PackagedDatasetModuleFactory(path, data_dir=data_dir, data_files=data_files, download_config=download_config, download_mode=download_mode).get_module()
    elif path.endswith(filename):
        if os.path.isfile(path):
            return LocalDatasetModuleFactoryWithScript(path, download_mode=download_mode, dynamic_modules_path=dynamic_modules_path, trust_remote_code=trust_remote_code).get_module()
        else:
            raise FileNotFoundError(f"Couldn't find a dataset script at {relative_to_absolute_path(path)}")
    elif os.path.isfile(combined_path):
        return LocalDatasetModuleFactoryWithScript(combined_path, download_mode=download_mode, dynamic_modules_path=dynamic_modules_path, trust_remote_code=trust_remote_code).get_module()
    elif os.path.isdir(path):
        return LocalDatasetModuleFactoryWithoutScript(path, data_dir=data_dir, data_files=data_files, download_mode=download_mode).get_module()
    elif is_relative_path(path) and path.count('/') <= 1:
        try:
            _raise_if_offline_mode_is_enabled()
            hf_api = HfApi(config.HF_ENDPOINT)
            try:
                dataset_info = hf_api.dataset_info(repo_id=path, revision=revision, token=download_config.token, timeout=100.0)
            except Exception as e:
                if isinstance(e, (OfflineModeIsEnabled, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError)):
                    raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({type(e).__name__})")
                elif '404' in str(e):
                    msg = f"Dataset '{path}' doesn't exist on the Hub or cannot be accessed"
                    raise DatasetNotFoundError(msg + f" at revision '{revision}'" if revision else msg)
                elif '401' in str(e):
                    msg = f"Dataset '{path}' doesn't exist on the Hub or cannot be accessed"
                    msg = msg + f" at revision '{revision}'" if revision else msg
                    raise DatasetNotFoundError(msg + f'. If the dataset is private or gated, make sure to log in with `huggingface-cli login` or visit the dataset page at https://huggingface.co/datasets/{path} to ask for access.')
                else:
                    raise e
            if filename in [sibling.rfilename for sibling in dataset_info.siblings]:
                fs = HfFileSystem(endpoint=config.HF_ENDPOINT, token=download_config.token)
                if _require_custom_configs or (revision and revision != 'main'):
                    can_load_config_from_parquet_export = False
                elif _require_default_config_name:
                    with fs.open(f'datasets/{path}/{filename}', 'r', encoding='utf-8') as f:
                        can_load_config_from_parquet_export = 'DEFAULT_CONFIG_NAME' not in f.read()
                else:
                    can_load_config_from_parquet_export = True
                if config.USE_PARQUET_EXPORT and can_load_config_from_parquet_export:
                    try:
                        return HubDatasetModuleFactoryWithParquetExport(path, download_config=download_config, revision=dataset_info.sha).get_module()
                    except _datasets_server.DatasetsServerError:
                        pass
                return HubDatasetModuleFactoryWithScript(path, revision=revision, download_config=download_config, download_mode=download_mode, dynamic_modules_path=dynamic_modules_path, trust_remote_code=trust_remote_code).get_module()
            else:
                return HubDatasetModuleFactoryWithoutScript(path, revision=revision, data_dir=data_dir, data_files=data_files, download_config=download_config, download_mode=download_mode).get_module()
        except Exception as e1:
            try:
                return CachedDatasetModuleFactory(path, dynamic_modules_path=dynamic_modules_path, cache_dir=cache_dir).get_module()
            except Exception:
                if isinstance(e1, OfflineModeIsEnabled):
                    raise ConnectionError(f"Couldn't reach the Hugging Face Hub for dataset '{path}': {e1}") from None
                if isinstance(e1, (DataFilesNotFoundError, DatasetNotFoundError, EmptyDatasetError)):
                    raise e1 from None
                if isinstance(e1, FileNotFoundError):
                    raise FileNotFoundError(f"Couldn't find a dataset script at {relative_to_absolute_path(combined_path)} or any data file in the same directory. Couldn't find '{path}' on the Hugging Face Hub either: {type(e1).__name__}: {e1}") from None
                raise e1 from None
    else:
        raise FileNotFoundError(f"Couldn't find a dataset script at {relative_to_absolute_path(combined_path)} or any data file in the same directory.")