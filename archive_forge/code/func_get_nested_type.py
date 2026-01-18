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
def get_nested_type(schema: FeatureType) -> pa.DataType:
    """
    get_nested_type() converts a datasets.FeatureType into a pyarrow.DataType, and acts as the inverse of
        generate_from_arrow_type().

    It performs double-duty as the implementation of Features.type and handles the conversion of
        datasets.Feature->pa.struct
    """
    if isinstance(schema, Features):
        return pa.struct({key: get_nested_type(schema[key]) for key in schema})
    elif isinstance(schema, dict):
        return pa.struct({key: get_nested_type(schema[key]) for key in schema})
    elif isinstance(schema, (list, tuple)):
        if len(schema) != 1:
            raise ValueError('When defining list feature, you should just provide one example of the inner type')
        value_type = get_nested_type(schema[0])
        return pa.list_(value_type)
    elif isinstance(schema, Sequence):
        value_type = get_nested_type(schema.feature)
        if isinstance(schema.feature, dict):
            return pa.struct({f.name: pa.list_(f.type, schema.length) for f in value_type})
        return pa.list_(value_type, schema.length)
    return schema()