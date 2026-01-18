import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
def is_numeric_feature(feature):
    from .. import ClassLabel, Sequence, Value
    from ..features.features import _ArrayXD
    if isinstance(feature, Sequence):
        return is_numeric_feature(feature.feature)
    elif isinstance(feature, list):
        return is_numeric_feature(feature[0])
    elif isinstance(feature, _ArrayXD):
        return is_numeric_pa_type(feature().storage_dtype)
    elif isinstance(feature, Value):
        return is_numeric_pa_type(feature())
    elif isinstance(feature, ClassLabel):
        return True
    else:
        return False