from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
def calculate_quantization_grid(values, border_count, border_type='Median'):
    assert border_count > 0, 'Border count should be > 0'
    return _calculate_quantization_grid(values, border_count, border_type)