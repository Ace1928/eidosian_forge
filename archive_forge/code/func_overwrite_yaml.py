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
def overwrite_yaml(root, file_name, data, ensure_yaml_extension=True):
    """Safely overwrites a preexisting yaml file, ensuring that file contents are not deleted or
    corrupted if the write fails. This is achieved by writing contents to a temporary file
    and moving the temporary file to replace the preexisting file, rather than opening the
    preexisting file for a direct write.

    Args:
        root: Directory name.
        file_name: File name.
        data: The data to write, represented as a dictionary.
        ensure_yaml_extension: If True, Will automatically add .yaml extension if not given.

    """
    tmp_file_path = None
    original_file_path = os.path.join(root, file_name)
    original_file_mode = os.stat(original_file_path).st_mode
    try:
        tmp_file_fd, tmp_file_path = tempfile.mkstemp(suffix='file.yaml')
        os.close(tmp_file_fd)
        write_yaml(root=get_parent_dir(tmp_file_path), file_name=os.path.basename(tmp_file_path), data=data, overwrite=True, sort_keys=True, ensure_yaml_extension=ensure_yaml_extension)
        shutil.move(tmp_file_path, original_file_path)
        os.chmod(original_file_path, original_file_mode)
    finally:
        if tmp_file_path is not None and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)