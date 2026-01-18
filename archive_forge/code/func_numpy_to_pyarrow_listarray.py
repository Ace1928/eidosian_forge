import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
import pyarrow_hotfix  # noqa: F401  # to fix vulnerability on pyarrow<14.0.1
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages
def numpy_to_pyarrow_listarray(arr: np.ndarray, type: pa.DataType=None) -> pa.ListArray:
    """Build a PyArrow ListArray from a multidimensional NumPy array"""
    arr = np.array(arr)
    values = pa.array(arr.flatten(), type=type)
    for i in range(arr.ndim - 1):
        n_offsets = reduce(mul, arr.shape[:arr.ndim - i - 1], 1)
        step_offsets = arr.shape[arr.ndim - i - 1]
        offsets = pa.array(np.arange(n_offsets + 1) * step_offsets, type=pa.int32())
        values = pa.ListArray.from_arrays(offsets, values)
    return values