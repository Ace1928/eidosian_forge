import base64
import sys
from typing import Iterable, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator
def get_facet_type_from_numpy_type(dtype):
    """Converts a Numpy dtype to the FeatureNameStatistics.Type proto enum."""
    fs_proto = facet_feature_statistics_pb2.FeatureNameStatistics
    if dtype.char in np.typecodes['Complex']:
        raise MlflowException('Found type complex, but expected one of: int, long, float, string, bool')
    elif dtype.char in np.typecodes['AllFloat']:
        return fs_proto.FLOAT
    elif dtype.char in np.typecodes['AllInteger'] or np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64):
        return fs_proto.INT
    else:
        return fs_proto.STRING