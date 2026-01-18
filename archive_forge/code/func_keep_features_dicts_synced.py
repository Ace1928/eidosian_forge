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
def keep_features_dicts_synced(func):
    """
    Wrapper to keep the secondary dictionary, which tracks whether keys are decodable, of the :class:`datasets.Features` object
    in sync with the main dictionary.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: 'Features' = args[0]
            args = args[1:]
        else:
            self: 'Features' = kwargs.pop('self')
        out = func(self, *args, **kwargs)
        assert hasattr(self, '_column_requires_decoding')
        self._column_requires_decoding = {col: require_decoding(feature) for col, feature in self.items()}
        return out
    wrapper._decorator_name_ = '_keep_dicts_synced'
    return wrapper