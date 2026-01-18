import importlib
import logging
import os
import pathlib
import posixpath
import sys
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import (
from mlflow.utils._spark_utils import (
from mlflow.utils.file_utils import (
def resolve_to_parquet(self, dst_path: str):
    if self.location is None and self.sql is None:
        raise MlflowException('Either location or sql configuration key must be specified for dataset with format spark_sql') from None
    spark_session = self._get_or_create_spark_session()
    spark_df = None
    if self.sql is not None:
        spark_df = spark_session.sql(self.sql)
    elif self.location is not None:
        spark_df = spark_session.table(self.location)
    pandas_df = self._convert_spark_df_to_pandas(spark_df)
    write_pandas_df_as_parquet(df=pandas_df, data_parquet_path=dst_path)