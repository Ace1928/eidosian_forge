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
def rename_columns(self, column_mapping: Dict[str, str]) -> 'IterableDatasetDict':
    """
        Rename several columns in the dataset, and move the features associated to the original columns under
        the new column names.
        The renaming is applied to all the datasets of the dataset dictionary.

        Args:
            column_mapping (`Dict[str, str]`):
                A mapping of columns to rename to their new names.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset with renamed columns

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> ds = ds.rename_columns({"text": "movie_review", "label": "rating"})
        >>> next(iter(ds["train"]))
        {'movie_review': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
         'rating': 1}
        ```
        """
    return IterableDatasetDict({k: dataset.rename_columns(column_mapping=column_mapping) for k, dataset in self.items()})