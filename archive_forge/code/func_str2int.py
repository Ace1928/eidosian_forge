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
def str2int(self, values: Union[str, Iterable]) -> Union[int, Iterable]:
    """Conversion class name `string` => `integer`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> ds.features["label"].str2int('neg')
        0
        ```
        """
    if not isinstance(values, str) and (not isinstance(values, Iterable)):
        raise ValueError(f'Values {values} should be a string or an Iterable (list, numpy array, pytorch, tensorflow tensors)')
    return_list = True
    if isinstance(values, str):
        values = [values]
        return_list = False
    output = [self._strval2int(value) for value in values]
    return output if return_list else output[0]