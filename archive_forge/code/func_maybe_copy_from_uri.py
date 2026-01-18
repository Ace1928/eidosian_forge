import logging
import os
import posixpath
import re
import shutil
from typing import Any, Dict, Optional
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import environment_variables, mleap, pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _LOG_MODEL_INFER_SIGNATURE_WARNING_TEMPLATE
from mlflow.models.utils import _Example, _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import (
from mlflow.utils import _get_fully_qualified_class_name, databricks_utils
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import (
@classmethod
def maybe_copy_from_uri(cls, src_uri, dst_path, local_model_path=None):
    """
        Conditionally copy the file to the Hadoop DFS from the source uri.
        In case the file is already on the Hadoop DFS do nothing.

        Returns:
            If copied, return new target location, otherwise return source uri.
        """
    try:
        dfs_path = cls._fs().makeQualified(cls._remote_path(src_uri))
        if cls._try_file_exists(dfs_path):
            _logger.info("File '%s' is already on DFS, copy is not necessary.", src_uri)
            return src_uri
    except Exception:
        _logger.info("URI '%s' does not point to the current DFS.", src_uri)
    _logger.info("File '%s' not found on DFS. Will attempt to upload the file.", src_uri)
    return cls.maybe_copy_from_local_file(local_model_path or _download_artifact_from_uri(src_uri), dst_path)