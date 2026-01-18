import json
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from packaging.version import Version
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.digest_utils import get_normalized_md5_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
def load_delta(path: Optional[str]=None, table_name: Optional[str]=None, version: Optional[str]=None, targets: Optional[str]=None, name: Optional[str]=None, digest: Optional[str]=None) -> SparkDataset:
    """
    Loads a :py:class:`SparkDataset <mlflow.data.spark_dataset.SparkDataset>` from a Delta table
    for use with MLflow Tracking.

    Args:
        path: The path to the Delta table. Either ``path`` or ``table_name`` must be specified.
        table_name: The name of the Delta table. Either ``path`` or ``table_name`` must be
            specified.
        version: The Delta table version. If not specified, the version will be inferred.
        targets: Optional. The name of the Delta table column containing targets (labels) for
            supervised learning.
        name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
            automatically generated.
        digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
            is automatically computed.

    Returns:
        An instance of :py:class:`SparkDataset <mlflow.data.spark_dataset.SparkDataset>`.
    """
    from mlflow.data.spark_delta_utils import _try_get_delta_table_latest_version_from_path, _try_get_delta_table_latest_version_from_table_name
    if (path, table_name).count(None) != 1:
        raise MlflowException('Must specify exactly one of `table_name` or `path`.', INVALID_PARAMETER_VALUE)
    if version is None:
        if path is not None:
            version = _try_get_delta_table_latest_version_from_path(path)
        else:
            version = _try_get_delta_table_latest_version_from_table_name(table_name)
    if name is None and table_name is not None:
        name = table_name + (f'@v{version}' if version is not None else '')
    source = DeltaDatasetSource(path=path, delta_table_name=table_name, delta_table_version=version)
    df = source.load()
    return SparkDataset(df=df, source=source, targets=targets, name=name, digest=digest)