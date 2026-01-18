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
def class_encode_column(self, column: str, include_nulls: bool=False) -> 'DatasetDict':
    """Casts the given column as [`~datasets.features.ClassLabel`] and updates the tables.

        Args:
            column (`str`):
                The name of the column to cast.
            include_nulls (`bool`, defaults to `False`):
                Whether to include null values in the class labels. If `True`, the null values will be encoded as the `"None"` class label.

                <Added version="1.14.2"/>

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("boolq")
        >>> ds["train"].features
        {'answer': Value(dtype='bool', id=None),
         'passage': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None)}
        >>> ds = ds.class_encode_column("answer")
        >>> ds["train"].features
        {'answer': ClassLabel(num_classes=2, names=['False', 'True'], id=None),
         'passage': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None)}
        ```
        """
    self._check_values_type()
    return DatasetDict({k: dataset.class_encode_column(column=column, include_nulls=include_nulls) for k, dataset in self.items()})