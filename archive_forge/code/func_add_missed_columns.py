from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
def add_missed_columns(start_column_idx, end_column_idx, non_feature_column_count):
    for missed_column_idx in range(start_column_idx, end_column_idx):
        column_name = 'feature_%i' % (missed_column_idx - non_feature_column_count)
        column_names.append(column_name)
        column_type_to_indices.setdefault('Num', []).append(missed_column_idx)
        column_dtypes[column_name] = np.float32