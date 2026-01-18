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
@experimental
def generate_signature_output(pipeline, data, model_config=None, params=None, flavor_config=None):
    """
    Utility for generating the response output for the purposes of extracting an output signature
    for model saving and logging. This function simulates loading of a saved model or pipeline
    as a ``pyfunc`` model without having to incur a write to disk.

    Args:
        pipeline: A ``transformers`` pipeline object. Note that component-level or model-level
            inputs are not permitted for extracting an output example.
        data: An example input that is compatible with the given pipeline
        model_config: Any additional model configuration, provided as kwargs, to inform
            the format of the output type from a pipeline inference call.
        params: A dictionary of additional parameters to pass to the pipeline for inference.
        flavor_config: The flavor configuration for the model.

    Returns:
        The output from the ``pyfunc`` pipeline wrapper's ``predict`` method
    """
    import transformers
    from mlflow.transformers import signature
    if not isinstance(pipeline, transformers.Pipeline):
        raise MlflowException(f'The pipeline type submitted is not a valid transformers Pipeline. The type {type(pipeline).__name__} is not supported.', error_code=INVALID_PARAMETER_VALUE)
    return signature.generate_signature_output(pipeline, data, model_config, params)