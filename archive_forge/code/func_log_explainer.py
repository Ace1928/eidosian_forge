import os
import tempfile
import types
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional
import numpy as np
import yaml
import mlflow
import mlflow.utils.autologging_utils
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_package_name
from mlflow.utils.uri import append_to_uri_path
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_explainer(explainer, artifact_path, serialize_model_using_mlflow=True, conda_env=None, code_paths=None, registered_model_name=None, signature: ModelSignature=None, input_example: ModelInputExample=None, await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS, pip_requirements=None, extra_pip_requirements=None, metadata=None):
    """
    Log an SHAP explainer as an MLflow artifact for the current run.

    Args:
        explainer: SHAP explainer to be saved.
        artifact_path: Run-relative artifact path.
        serialize_model_using_mlflow: When set to True, MLflow will extract the underlying
            model and serialize it as an MLmodel, otherwise it uses SHAP's internal serialization.
            Defaults to True. Currently MLflow serialization is only supported for models of
            'sklearn' or 'pytorch' flavors.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under ``registered_model_name``,
            also creating a registered model if one with the given name does not exist.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>` describes model input
            and output :py:class:`Schema <mlflow.types.Schema>`. The model signature can be
            :py:func:`inferred <mlflow.models.infer_signature>` from datasets with valid model input
            (e.g. the training dataset with target column omitted) and valid model output
            (e.g. model predictions generated on the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                train = df.drop_column("target_label")
                predictions = ...  # compute model predictions
                signature = infer_signature(train, predictions)
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function waits for five
            minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.shap, explainer=explainer, conda_env=conda_env, code_paths=code_paths, serialize_model_using_mlflow=serialize_model_using_mlflow, registered_model_name=registered_model_name, signature=signature, input_example=input_example, await_registration_for=await_registration_for, pip_requirements=pip_requirements, extra_pip_requirements=extra_pip_requirements, metadata=metadata)