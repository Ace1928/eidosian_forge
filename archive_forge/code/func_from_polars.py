import contextlib
import copy
import fnmatch
import itertools
import json
import math
import os
import posixpath
import re
import shutil
import sys
import tempfile
import time
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import partial, wraps
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from random import sample
from typing import (
from typing import Sequence as Sequence_
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from fsspec.core import url_to_fs
from huggingface_hub import (
from huggingface_hub.hf_api import RepoFile
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter, OptimizedTypedSequence
from .data_files import sanitize_patterns
from .download.streaming_download_manager import xgetsize
from .features import Audio, ClassLabel, Features, Image, Sequence, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .fingerprint import (
from .formatting import format_table, get_format_type_from_alias, get_formatter, query_table
from .formatting.formatting import LazyDict, _is_range_contiguous
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .search import IndexableMixin
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import (
from .tasks import TaskTemplate
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.deprecation_utils import deprecated
from .utils.file_utils import estimate_dataset_size
from .utils.info_utils import is_small_dataset
from .utils.metadata import MetadataConfigs
from .utils.py_utils import (
from .utils.stratify import stratified_shuffle_split_generate_indices
from .utils.tf_utils import dataset_to_tf, minimal_tf_collate_fn, multiprocess_dataset_to_tf
from .utils.typing import ListLike, PathLike
@classmethod
def from_polars(cls, df: 'pl.DataFrame', features: Optional[Features]=None, info: Optional[DatasetInfo]=None, split: Optional[NamedSplit]=None) -> 'Dataset':
    """
        Collect the underlying arrow arrays in an Arrow Table.

        This operation is mostly zero copy.

        Data types that do copy:
            * CategoricalType

        Args:
            df (`polars.DataFrame`): DataFrame to convert to Arrow Table
            features (`Features`, optional): Dataset features.
            info (`DatasetInfo`, optional): Dataset information, like description, citation, etc.
            split (`NamedSplit`, optional): Name of the dataset split.

        Examples:
        ```py
        >>> ds = Dataset.from_polars(df)
        ```
        """
    if info is not None and features is not None and (info.features != features):
        raise ValueError(f"Features specified in `features` and `info.features` can't be different:\n{features}\n{info.features}")
    features = features if features is not None else info.features if info is not None else None
    if info is None:
        info = DatasetInfo()
    info.features = features
    table = InMemoryTable(df.to_arrow())
    if features is not None:
        table = table.cast(features.arrow_schema)
    return cls(table, info=info, split=split)