import base64
import sys
from typing import Iterable, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator
def compute_common_stats(column) -> facet_feature_statistics_pb2.CommonStatistics:
    """
    Computes common statistics for a given column in the DataFrame.

    Args:
        column: A column from a DataFrame.

    Returns:
        A CommonStatistics proto.
    """
    common_stats = facet_feature_statistics_pb2.CommonStatistics()
    common_stats.num_missing = column.isnull().sum()
    common_stats.num_non_missing = len(column) - common_stats.num_missing
    common_stats.min_num_values = 1
    common_stats.max_num_values = 1
    common_stats.avg_num_values = 1.0
    return common_stats