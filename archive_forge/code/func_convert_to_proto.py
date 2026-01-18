import base64
import sys
from typing import Iterable, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator
def convert_to_proto(df: pd.DataFrame) -> facet_feature_statistics_pb2.DatasetFeatureStatisticsList:
    """
    Converts the data from DataFrame format to DatasetFeatureStatisticsList proto.

    Args:
        df: The DataFrame for which feature statistics need to be computed.

    Returns:
        A DatasetFeatureStatisticsList proto.
    """
    feature_stats = convert_to_dataset_feature_statistics(df)
    feature_stats_list = facet_feature_statistics_pb2.DatasetFeatureStatisticsList()
    feature_stats_list.datasets.append(feature_stats)
    return feature_stats_list