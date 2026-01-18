import atexit
import codecs
import errno
import fnmatch
import gzip
import json
import logging
import math
import os
import pathlib
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import CalledProcessError, TimeoutExpired
from typing import Optional, Union
from urllib.parse import unquote
from urllib.request import pathname2url
import yaml
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MissingConfigException, MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialType
from mlflow.utils import download_cloud_file_chunk, merge_dicts
from mlflow.utils.databricks_utils import _get_dbutils
from mlflow.utils.os import is_windows
from mlflow.utils.process import cache_return_value_per_process
from mlflow.utils.request_utils import cloud_storage_http_request, download_chunk
from mlflow.utils.rest_utils import augmented_raise_for_status
def read_parquet_as_pandas_df(data_parquet_path: str):
    """Deserialize and load the specified parquet file as a Pandas DataFrame.

    Args:
        data_parquet_path: String, path object (implementing os.PathLike[str]),
        or file-like object implementing a binary read() function. The string
        could be a URL. Valid URL schemes include http, ftp, s3, gs, and file.
        For file URLs, a host is expected. A local file could
        be: file://localhost/path/to/table.parquet. A file URL can also be a path to a
        directory that contains multiple partitioned parquet files. Pyarrow
        support paths to directories as well as file URLs. A directory
        path could be: file://localhost/path/to/tables or s3://bucket/partition_dir.

    Returns:
        pandas dataframe
    """
    import pandas as pd
    return pd.read_parquet(data_parquet_path, engine='pyarrow')