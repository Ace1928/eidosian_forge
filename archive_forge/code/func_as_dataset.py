import abc
import contextlib
import copy
import inspect
import os
import posixpath
import shutil
import textwrap
import time
import urllib
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple, Union
from unittest.mock import patch
import fsspec
import pyarrow as pa
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config, utils
from .arrow_dataset import Dataset
from .arrow_reader import (
from .arrow_writer import ArrowWriter, BeamWriter, ParquetWriter, SchemaInferenceError
from .data_files import DataFilesDict, DataFilesPatternsDict, sanitize_patterns
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager, DownloadMode
from .download.mock_download_manager import MockDownloadManager
from .download.streaming_download_manager import StreamingDownloadManager, xjoin, xopen
from .exceptions import DatasetGenerationCastError, DatasetGenerationError, FileFormatError, ManualDownloadError
from .features import Features
from .filesystems import (
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict, PostProcessedInfo
from .iterable_dataset import ArrowExamplesIterable, ExamplesIterable, IterableDataset
from .keyhash import DuplicatedKeysError
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase
from .splits import Split, SplitDict, SplitGenerator, SplitInfo
from .streaming import extend_dataset_builder_for_streaming
from .table import CastError
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils._filelock import FileLock
from .utils.file_utils import cached_path, is_remote_url
from .utils.info_utils import VerificationMode, get_size_checksum_dict, verify_checksums, verify_splits
from .utils.py_utils import (
from .utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
from .utils.track import tracked_list
def as_dataset(self, split: Optional[Split]=None, run_post_process=True, verification_mode: Optional[Union[VerificationMode, str]]=None, ignore_verifications='deprecated', in_memory=False) -> Union[Dataset, DatasetDict]:
    """Return a Dataset for the specified split.

        Args:
            split (`datasets.Split`):
                Which subset of the data to return.
            run_post_process (`bool`, defaults to `True`):
                Whether to run post-processing dataset transforms and/or add
                indexes.
            verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
                Verification mode determining the checks to run on the
                downloaded/processed dataset information (checksums/size/splits/...).

                <Added version="2.9.1"/>
            ignore_verifications (`bool`, defaults to `False`):
                Whether to ignore the verifications of the
                downloaded/processed dataset information (checksums/size/splits/...).

                <Deprecated version="2.9.1">

                `ignore_verifications` was deprecated in version 2.9.1 and will be removed in 3.0.0.
                Please use `verification_mode` instead.

                </Deprecated>
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            datasets.Dataset

        Example:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder('rotten_tomatoes')
        >>> builder.download_and_prepare()
        >>> ds = builder.as_dataset(split='train')
        >>> ds
        Dataset({
            features: ['text', 'label'],
            num_rows: 8530
        })
        ```
        """
    if ignore_verifications != 'deprecated':
        verification_mode = verification_mode.NO_CHECKS if ignore_verifications else VerificationMode.ALL_CHECKS
        warnings.warn(f"'ignore_verifications' was deprecated in favor of 'verification' in version 2.9.1 and will be removed in 3.0.0.\nYou can remove this warning by passing 'verification_mode={verification_mode.value}' instead.", FutureWarning)
    if self._file_format is not None and self._file_format != 'arrow':
        raise FileFormatError('Loading a dataset not written in the "arrow" format is not supported.')
    if is_remote_filesystem(self._fs):
        raise NotImplementedError(f'Loading a dataset cached in a {type(self._fs).__name__} is not supported.')
    if not os.path.exists(self._output_dir):
        raise FileNotFoundError(f'Dataset {self.dataset_name}: could not find data in {self._output_dir}. Please make sure to call builder.download_and_prepare(), or use datasets.load_dataset() before trying to access the Dataset object.')
    logger.debug(f'Constructing Dataset for split {split or ', '.join(self.info.splits)}, from {self._output_dir}')
    if split is None:
        split = {s: s for s in self.info.splits}
    verification_mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)
    datasets = map_nested(partial(self._build_single_dataset, run_post_process=run_post_process, verification_mode=verification_mode, in_memory=in_memory), split, map_tuple=True, disable_tqdm=True)
    if isinstance(datasets, dict):
        datasets = DatasetDict(datasets)
    return datasets