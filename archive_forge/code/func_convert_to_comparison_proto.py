import base64
import sys
from typing import Iterable, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator
def convert_to_comparison_proto(dfs: Iterable[Tuple[str, pd.DataFrame]]) -> facet_feature_statistics_pb2.DatasetFeatureStatisticsList:
    """
    Converts a collection of named stats DataFrames to a single DatasetFeatureStatisticsList proto.

    Args:
        dfs: The named "glimpses" that contain the DataFrame. Each "glimpse"
            DataFrame has the same properties as the input to `convert_to_proto()`.

    Returns:
        A DatasetFeatureStatisticsList proto which contains a translation
        of the glimpses with the given names.
    """
    feature_stats_list = facet_feature_statistics_pb2.DatasetFeatureStatisticsList()
    for name, df in dfs:
        if not df.empty:
            proto = convert_to_dataset_feature_statistics(df)
            proto.name = name
            feature_stats_list.datasets.append(proto)
    return feature_stats_list