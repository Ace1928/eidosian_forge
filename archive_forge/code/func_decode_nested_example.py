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
def decode_nested_example(schema, obj, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]]=None):
    """Decode a nested example.
    This is used since some features (in particular Audio and Image) have some logic during decoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be decoded.
    If the first element needs to be decoded, then all the elements of the list will be decoded, otherwise they'll stay the same.
    """
    if isinstance(schema, dict):
        return {k: decode_nested_example(sub_schema, sub_obj) for k, (sub_schema, sub_obj) in zip_dict(schema, obj)} if obj is not None else None
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if decode_nested_example(sub_schema, first_elmt) != first_elmt:
                    return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, Sequence):
        if isinstance(schema.feature, dict):
            return {k: decode_nested_example([schema.feature[k]], obj[k]) for k in schema.feature}
        else:
            return decode_nested_example([schema.feature], obj)
    elif isinstance(schema, (Audio, Image)):
        if obj is not None and schema.decode:
            return schema.decode_example(obj, token_per_repo_id=token_per_repo_id)
    return obj