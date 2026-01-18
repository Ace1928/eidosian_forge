import inspect
import logging
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, get_type_hints
import numpy as np
import pandas as pd
from mlflow import environment_variables
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _contains_params, _Example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _upload_artifact_to_uri
from mlflow.types.schema import ParamSchema, Schema
from mlflow.types.utils import _infer_param_schema, _infer_schema, _infer_schema_from_type_hint
from mlflow.utils.uri import append_to_uri_path
def infer_signature(model_input: Any=None, model_output: 'MlflowInferableDataset'=None, params: Optional[Dict[str, Any]]=None) -> ModelSignature:
    """
    Infer an MLflow model signature from the training data (input), model predictions (output)
    and parameters (for inference).

    The signature represents model input and output as data frames with (optionally) named columns
    and data type specified as one of types defined in :py:class:`mlflow.types.DataType`. It also
    includes parameters schema for inference, .
    This method will raise an exception if the user data contains incompatible types or is not
    passed in one of the supported formats listed below.

    The input should be one of these:
      - pandas.DataFrame
      - pandas.Series
      - dictionary of { name -> numpy.ndarray}
      - numpy.ndarray
      - pyspark.sql.DataFrame
      - scipy.sparse.csr_matrix
      - scipy.sparse.csc_matrix
      - dictionary / list of dictionaries of JSON-convertible types

    The element types should be mappable to one of :py:class:`mlflow.types.DataType`.

    For pyspark.sql.DataFrame inputs, columns of type DateType and TimestampType are both inferred
    as type :py:data:`datetime <mlflow.types.DataType.datetime>`, which is coerced to
    TimestampType at inference.

    Args:
      model_input: Valid input to the model. E.g. (a subset of) the training dataset.
      model_output: Valid model output. E.g. Model predictions for the (subset of) training
                    dataset.
      params: Valid parameters for inference. It should be a dictionary of parameters
              that can be set on the model during inference by passing `params` to pyfunc
              `predict` method.

              An example of valid parameters:

              .. code-block:: python

                    from mlflow.models import infer_signature
                    from mlflow.transformers import generate_signature_output

                    # Define parameters for inference
                    params = {
                        "num_beams": 5,
                        "max_length": 30,
                        "do_sample": True,
                        "remove_invalid_values": True,
                    }

                    # Infer the signature including parameters
                    signature = infer_signature(
                        data,
                        generate_signature_output(model, data),
                        params=params,
                    )

                    # Saving model with model signature
                    mlflow.transformers.save_model(
                        model,
                        path=model_path,
                        signature=signature,
                    )

                    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

                    # Passing params to `predict` function directly
                    result = pyfunc_loaded.predict(data, params=params)

              .. Note:: Experimental: This parameter may change or be removed in a future
                                     release without warning.

    Returns:
      ModelSignature
    """
    inputs = _infer_schema(model_input) if model_input is not None else None
    outputs = _infer_schema(model_output) if model_output is not None else None
    params = _infer_param_schema(params) if params else None
    return ModelSignature(inputs, outputs, params)