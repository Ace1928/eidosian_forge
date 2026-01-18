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
def int2str(self, values: Union[int, Iterable]) -> Union[str, Iterable]:
    """Conversion `integer` => class name `string`.

        Regarding unknown/missing labels: passing negative integers raises `ValueError`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> ds.features["label"].int2str(0)
        'neg'
        ```
        """
    if not isinstance(values, int) and (not isinstance(values, Iterable)):
        raise ValueError(f'Values {values} should be an integer or an Iterable (list, numpy array, pytorch, tensorflow tensors)')
    return_list = True
    if isinstance(values, int):
        values = [values]
        return_list = False
    for v in values:
        if not 0 <= v < self.num_classes:
            raise ValueError(f'Invalid integer class label {v:d}')
    output = [self._int2str[int(v)] for v in values]
    return output if return_list else output[0]