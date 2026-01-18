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
def list_of_np_array_to_pyarrow_listarray(l_arr: List[np.ndarray], type: pa.DataType=None) -> pa.ListArray:
    """Build a PyArrow ListArray from a possibly nested list of NumPy arrays"""
    if len(l_arr) > 0:
        return list_of_pa_arrays_to_pyarrow_listarray([numpy_to_pyarrow_listarray(arr, type=type) if arr is not None else None for arr in l_arr])
    else:
        return pa.array([], type=type)