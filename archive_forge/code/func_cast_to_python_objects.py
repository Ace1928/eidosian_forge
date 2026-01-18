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
def cast_to_python_objects(obj: Any, only_1d_for_numpy=False, optimize_list_casting=True) -> Any:
    """
    Cast numpy/pytorch/tensorflow/pandas objects to python lists.
    It works recursively.

    If `optimize_list_casting` is True, To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be casted.
    If the first element needs to be casted, then all the elements of the list will be casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating over every single token for example.

    Args:
        obj: the object (nested struct) to cast
        only_1d_for_numpy (bool, default ``False``): whether to keep the full multi-dim tensors as multi-dim numpy arrays, or convert them to
            nested lists of 1-dimensional numpy arrays. This can be useful to keep only 1-d arrays to instantiate Arrow arrays.
            Indeed Arrow only support converting 1-dimensional array values.
        optimize_list_casting (bool, default ``True``): whether to optimize list casting by checking the first non-null element to see if it needs to be casted
            and if it doesn't, not checking the rest of the list elements.

    Returns:
        casted_obj: the casted object
    """
    return _cast_to_python_objects(obj, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting)[0]