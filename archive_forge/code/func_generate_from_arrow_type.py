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
def generate_from_arrow_type(pa_type: pa.DataType) -> FeatureType:
    """
    generate_from_arrow_type accepts an arrow DataType and returns a datasets FeatureType to be used as the type for
        a single field.

    This is the high-level arrow->datasets type conversion and is inverted by get_nested_type().

    This operates at the individual *field* level, whereas Features.from_arrow_schema() operates at the
        full schema level and holds the methods that represent the bijection from Features<->pyarrow.Schema
    """
    if isinstance(pa_type, pa.StructType):
        return {field.name: generate_from_arrow_type(field.type) for field in pa_type}
    elif isinstance(pa_type, pa.FixedSizeListType):
        return Sequence(feature=generate_from_arrow_type(pa_type.value_type), length=pa_type.list_size)
    elif isinstance(pa_type, pa.ListType):
        feature = generate_from_arrow_type(pa_type.value_type)
        if isinstance(feature, (dict, tuple, list)):
            return [feature]
        return Sequence(feature=feature)
    elif isinstance(pa_type, _ArrayXDExtensionType):
        array_feature = [None, None, Array2D, Array3D, Array4D, Array5D][pa_type.ndims]
        return array_feature(shape=pa_type.shape, dtype=pa_type.value_type)
    elif isinstance(pa_type, pa.DictionaryType):
        raise NotImplementedError
    elif isinstance(pa_type, pa.DataType):
        return Value(dtype=_arrow_to_datasets_dtype(pa_type))
    else:
        raise ValueError(f'Cannot convert {pa_type} to a Feature type.')