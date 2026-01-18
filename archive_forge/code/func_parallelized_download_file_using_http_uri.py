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
def parallelized_download_file_using_http_uri(thread_pool_executor, http_uri, download_path, remote_file_path, file_size, uri_type, chunk_size, env, headers=None):
    """
    Downloads a file specified using the `http_uri` to a local `download_path`. This function
    sends multiple requests in parallel each specifying its own desired byte range as a header,
    then reconstructs the file from the downloaded chunks. This allows for downloads of large files
    without OOM risk.

    Note : This function is meant to download files using presigned urls from various cloud
            providers.
    Returns a dict of chunk index : exception, if one was thrown for that index.
    """

    def run_download(chunk: _Chunk):
        try:
            subprocess.run([sys.executable, download_cloud_file_chunk.__file__, '--range-start', str(chunk.start), '--range-end', str(chunk.end), '--headers', json.dumps(headers or {}), '--download-path', download_path, '--http-uri', http_uri], text=True, check=True, capture_output=True, timeout=MLFLOW_DOWNLOAD_CHUNK_TIMEOUT.get(), env=env)
        except (TimeoutExpired, CalledProcessError) as e:
            raise MlflowException(f'\n----- stdout -----\n{e.stdout.strip()}\n\n----- stderr -----\n{e.stderr.strip()}\n') from e
    chunks = _yield_chunks(remote_file_path, file_size, chunk_size)
    with open(download_path, 'w'):
        pass
    if uri_type == ArtifactCredentialType.GCP_SIGNED_URL or uri_type is None:
        chunk = next(chunks)
        download_chunk(range_start=chunk.start, range_end=chunk.end, headers=headers, download_path=download_path, http_uri=http_uri)
        downloaded_size = os.path.getsize(download_path)
        if downloaded_size > chunk_size:
            return {}
    futures = {thread_pool_executor.submit(run_download, chunk): chunk for chunk in chunks}
    failed_downloads = {}
    with ArtifactProgressBar.chunks(file_size, f'Downloading {download_path}', chunk_size) as pbar:
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                future.result()
            except Exception as e:
                _logger.debug(f'Failed to download chunk {chunk.index} for {chunk.path}: {e}. The download of this chunk will be retried later.')
                failed_downloads[chunk] = future.exception()
            else:
                pbar.update()
    return failed_downloads