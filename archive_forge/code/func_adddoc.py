import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from scipy.special import softmax
from ._typing import ArrayLike, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import SKLEARN_INSTALLED, XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .config import config_context
from .core import (
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array, _is_pandas_df
from .training import train
def adddoc(cls: Type) -> Type:
    doc = ['\nParameters\n----------\n']
    if extra_parameters:
        doc.append(extra_parameters)
    doc.extend([get_doc(i) for i in items])
    if end_note:
        doc.append(end_note)
    full_doc = [header + '\nSee :doc:`/python/sklearn_estimator` for more information.\n']
    full_doc.extend(doc)
    cls.__doc__ = ''.join(full_doc)
    return cls