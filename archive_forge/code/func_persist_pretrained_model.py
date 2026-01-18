from __future__ import annotations
import ast
import base64
import binascii
import contextlib
import copy
import functools
import importlib
import json
import logging
import os
import pathlib
import re
import shutil
import string
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
from mlflow import pyfunc
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _get_root_uri_and_artifact_path
from mlflow.transformers.flavor_config import (
from mlflow.transformers.hub_utils import is_valid_hf_repo_id
from mlflow.transformers.llm_inference_utils import (
from mlflow.transformers.model_io import (
from mlflow.transformers.peft import (
from mlflow.transformers.signature import (
from mlflow.transformers.torch_utils import _TORCH_DTYPE_KEY, _deserialize_torch_dtype
from mlflow.types.utils import _validate_input_dictionary_contains_only_strings_and_lists_of_strings
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import (
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.logging_utils import suppress_logs
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def persist_pretrained_model(model_uri: str) -> None:
    """
    Persist Transformers pretrained model weights to the artifacts directory of the specified
    model_uri. This API is primary used for updating an MLflow Model that was logged or saved
    with setting save_pretrained=False. Such models cannot be registered to Databricks Workspace
    Model Registry, due to the full pretrained model weights being absent in the artifacts.
    Transformers models saved in this mode store only the reference to the HuggingFace Hub
    repository. This API will download the model weights from the HuggingFace Hub repository
    and save them in the artifacts of the given model_uri so that the model can be registered
    to Databricks Workspace Model Registry.

    Args:
        model_uri: The URI of the existing MLflow Model of the Transformers flavor.
            It must be logged/saved with save_pretrained=False.

    Examples:

    .. code-block:: python

        import mlflow

        # Saving a model with save_pretrained=False
        with mlflow.start_run() as run:
            model = pipeline("question-answering", "csarron/mobilebert-uncased-squad-v2")
            mlflow.transformers.log_model(
                transformers_model=model, artifact_path="pipeline", save_pretrained=False
            )

        # The model cannot be registered to the Model Registry as it is
        try:
            mlflow.register_model(f"runs:/{run.info.run_id}/pipeline", "qa_pipeline")
        except MlflowException as e:
            print(e.message)

        # Use this API to persist the pretrained model weights
        mlflow.transformers.persist_pretrained_model(f"runs:/{run.info.run_id}/pipeline")

        # Now the model can be registered to the Model Registry
        mlflow.register_model(f"runs:/{run.info.run_id}/pipeline", "qa_pipeline")
    """
    root_uri, artifact_path = _get_root_uri_and_artifact_path(model_uri)
    artifact_repo = get_artifact_repository(root_uri)
    file_names = [os.path.basename(f.path) for f in artifact_repo.list_artifacts(artifact_path)]
    if MLMODEL_FILE_NAME in file_names and _MODEL_BINARY_FILE_NAME in file_names:
        _logger.info(f'The full pretrained model weight already exists in the artifact directory of the specified model_uri: {model_uri}. No action is needed.')
        return
    with TempDir() as tmp_dir:
        local_model_path = artifact_repo.download_artifacts(artifact_path, dst_path=tmp_dir.path())
        pipeline = load_model(local_model_path, return_type='pipeline')
        mlmodel_path = os.path.join(local_model_path, MLMODEL_FILE_NAME)
        model_conf = Model.load(mlmodel_path)
        updated_flavor_conf = update_flavor_conf_to_persist_pretrained_model(model_conf.flavors[FLAVOR_NAME])
        model_conf.add_flavor(FLAVOR_NAME, **updated_flavor_conf)
        model_conf.save(mlmodel_path)
        save_pipeline_pretrained_weights(pathlib.Path(local_model_path), pipeline, updated_flavor_conf)
        for dir_to_upload in (_MODEL_BINARY_FILE_NAME, _COMPONENTS_BINARY_DIR_NAME):
            local_dir = os.path.join(local_model_path, dir_to_upload)
            if not os.path.isdir(local_dir):
                continue
            try:
                artifact_repo.log_artifacts(local_dir, os.path.join(artifact_path, dir_to_upload))
            except Exception as e:
                raise MlflowException(f'Failed to upload {local_dir} to the existing model_uri due to {e}.Some other files may have been uploaded.') from e
        artifact_repo.log_artifact(mlmodel_path, artifact_path)
    _logger.info(f'The pretrained model has been successfully persisted in {model_uri}.')