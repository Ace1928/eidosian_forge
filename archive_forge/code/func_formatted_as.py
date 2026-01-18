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
@contextlib.contextmanager
def formatted_as(self, type: Optional[str]=None, columns: Optional[List]=None, output_all_columns: bool=False, **format_kwargs):
    """To be used in a `with` statement. Set `__getitem__` return format (type and columns).
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            type (`str`, *optional*):
                Output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']`.
                `None` means `__getitem__` returns python objects (default).
            columns (`List[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to False):
                Keep un-formatted columns as well in the output (as python objects).
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.
        """
    self._check_values_type()
    old_format_type = {k: dataset._format_type for k, dataset in self.items()}
    old_format_kwargs = {k: dataset._format_kwargs for k, dataset in self.items()}
    old_format_columns = {k: dataset._format_columns for k, dataset in self.items()}
    old_output_all_columns = {k: dataset._output_all_columns for k, dataset in self.items()}
    try:
        self.set_format(type, columns, output_all_columns, **format_kwargs)
        yield
    finally:
        for k, dataset in self.items():
            dataset.set_format(old_format_type[k], old_format_columns[k], old_output_all_columns[k], **old_format_kwargs[k])