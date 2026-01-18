import datetime
import importlib.metadata
import os
import posixpath
import urllib.parse
from collections import namedtuple
from packaging.version import Version
from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
from mlflow.environment_variables import (
from mlflow.exceptions import _UnsupportedMultipartUploadException
from mlflow.store.artifact.artifact_repo import ArtifactRepository, MultipartUploadMixin
from mlflow.utils.file_utils import relative_path_to_artifact_path
@staticmethod
def parse_gcs_uri(uri):
    """Parse an GCS URI, returning (bucket, path)"""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != 'gs':
        raise Exception(f'Not a GCS URI: {uri}')
    path = parsed.path
    if path.startswith('/'):
        path = path[1:]
    return (parsed.netloc, path)