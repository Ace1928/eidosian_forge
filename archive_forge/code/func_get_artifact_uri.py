import atexit
import contextlib
import importlib
import inspect
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking import _get_store, artifact_utils
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.default_experiment import registry as default_experiment_registry
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.annotations import experimental
from mlflow.utils.async_logging.run_operations import RunOperations
from mlflow.utils.autologging_utils import (
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.import_hooks import register_post_import_hook
from mlflow.utils.mlflow_tags import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import _validate_experiment_id_type, _validate_run_id
def get_artifact_uri(artifact_path: Optional[str]=None) -> str:
    """
    Get the absolute URI of the specified artifact in the currently active run.

    If `path` is not specified, the artifact root URI of the currently active
    run will be returned; calls to ``log_artifact`` and ``log_artifacts`` write
    artifact(s) to subdirectories of the artifact root URI.

    If no run is active, this method will create a new active run.

    Args:
        artifact_path: The run-relative artifact path for which to obtain an absolute URI.
            For example, "path/to/artifact". If unspecified, the artifact root URI
            for the currently active run will be returned.

    Returns:
        An *absolute* URI referring to the specified artifact or the currently active run's
        artifact root. For example, if an artifact path is provided and the currently active
        run uses an S3-backed store, this may be a uri of the form
        ``s3://<bucket_name>/path/to/artifact/root/path/to/artifact``. If an artifact path
        is not provided and the currently active run uses an S3-backed store, this may be a
        URI of the form ``s3://<bucket_name>/path/to/artifact/root``.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow

        features = "rooms, zipcode, median_price, school_rating, transport"
        with open("features.txt", "w") as f:
            f.write(features)

        # Log the artifact in a directory "features" under the root artifact_uri/features
        with mlflow.start_run():
            mlflow.log_artifact("features.txt", artifact_path="features")

            # Fetch the artifact uri root directory
            artifact_uri = mlflow.get_artifact_uri()
            print(f"Artifact uri: {artifact_uri}")

            # Fetch a specific artifact uri
            artifact_uri = mlflow.get_artifact_uri(artifact_path="features/features.txt")
            print(f"Artifact uri: {artifact_uri}")

    .. code-block:: text
        :caption: Output

        Artifact uri: file:///.../0/a46a80f1c9644bd8f4e5dd5553fffce/artifacts
        Artifact uri: file:///.../0/a46a80f1c9644bd8f4e5dd5553fffce/artifacts/features/features.txt
    """
    return artifact_utils.get_artifact_uri(run_id=_get_or_start_run().info.run_id, artifact_path=artifact_path)