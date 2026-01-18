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
def run_download(chunk: _Chunk):
    try:
        subprocess.run([sys.executable, download_cloud_file_chunk.__file__, '--range-start', str(chunk.start), '--range-end', str(chunk.end), '--headers', json.dumps(headers or {}), '--download-path', download_path, '--http-uri', http_uri], text=True, check=True, capture_output=True, timeout=MLFLOW_DOWNLOAD_CHUNK_TIMEOUT.get(), env=env)
    except (TimeoutExpired, CalledProcessError) as e:
        raise MlflowException(f'\n----- stdout -----\n{e.stdout.strip()}\n\n----- stderr -----\n{e.stderr.strip()}\n') from e