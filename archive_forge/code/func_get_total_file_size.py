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
def get_total_file_size(path: Union[str, pathlib.Path]) -> Optional[int]:
    """Return the size of all files under given path, including files in subdirectories.

    Args:
        path: The absolute path of a local directory.

    Returns:
        size in bytes.

    """
    try:
        if isinstance(path, pathlib.Path):
            path = str(path)
        if not os.path.exists(path):
            raise MlflowException(message=f'The given {path} does not exist.', error_code=INVALID_PARAMETER_VALUE)
        if not os.path.isdir(path):
            raise MlflowException(message=f'The given {path} is not a directory.', error_code=INVALID_PARAMETER_VALUE)
        total_size = 0
        for cur_path, dirs, files in os.walk(path):
            full_paths = [os.path.join(cur_path, file) for file in files]
            total_size += sum([os.path.getsize(file) for file in full_paths])
        return total_size
    except Exception as e:
        _logger.info(f'Failed to get the total size of {path} because of error :{e}')
        return None