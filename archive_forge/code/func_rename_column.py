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
def rename_column(self, original_column_name: str, new_column_name: str) -> 'IterableDatasetDict':
    """
        Rename a column in the dataset, and move the features associated to the original column under the new column
        name.
        The renaming is applied to all the datasets of the dataset dictionary.

        Args:
            original_column_name (`str`):
                Name of the column to rename.
            new_column_name (`str`):
                New name for the column.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset with a renamed column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> ds = ds.rename_column("text", "movie_review")
        >>> next(iter(ds["train"]))
        {'label': 1,
         'movie_review': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
    return IterableDatasetDict({k: dataset.rename_column(original_column_name=original_column_name, new_column_name=new_column_name) for k, dataset in self.items()})