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
def get_exported_dataset_info(self) -> DatasetInfo:
    """Empty `DatasetInfo` if doesn't exist

        Example:

        ```py
        >>> from datasets import load_dataset_builder
        >>> ds_builder = load_dataset_builder('rotten_tomatoes')
        >>> ds_builder.get_exported_dataset_info()
        DatasetInfo(description="Movie Review Dataset.
This is a dataset of containing 5,331 positive and 5,331 negative processed
sentences from Rotten Tomatoes movie reviews. This data was first used in Bo
Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for
sentiment categorization with respect to rating scales.'', Proceedings of the
ACL, 2005.
", citation='@InProceedings{Pang+Lee:05a,
  author =       {Bo Pang and Lillian Lee},
  title =        {Seeing stars: Exploiting class relationships for sentiment
                  categorization with respect to rating scales},
  booktitle =    {Proceedings of the ACL},
  year =         2005
}
', homepage='http://www.cs.cornell.edu/people/pabo/movie-review-data/', license='', features={'text': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None)}, post_processed=None, supervised_keys=SupervisedKeysData(input='', output=''), task_templates=[TextClassification(task='text-classification', text_column='text', label_column='label')], builder_name='rotten_tomatoes_movie_review', config_name='default', version=1.0.0, splits={'train': SplitInfo(name='train', num_bytes=1074810, num_examples=8530, dataset_name='rotten_tomatoes_movie_review'), 'validation': SplitInfo(name='validation', num_bytes=134679, num_examples=1066, dataset_name='rotten_tomatoes_movie_review'), 'test': SplitInfo(name='test', num_bytes=135972, num_examples=1066, dataset_name='rotten_tomatoes_movie_review')}, download_checksums={'https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz': {'num_bytes': 487770, 'checksum': 'a05befe52aafda71d458d188a1c54506a998b1308613ba76bbda2e5029409ce9'}}, download_size=487770, post_processing_size=None, dataset_size=1345461, size_in_bytes=1833231)
        ```
        """
    return self.get_all_exported_dataset_infos().get(self.config.name, DatasetInfo())