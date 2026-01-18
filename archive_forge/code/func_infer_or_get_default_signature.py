import json
import logging
import numpy as np
from mlflow.environment_variables import MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.models.utils import _contains_params
from mlflow.types.schema import ColSpec, DataType, Schema, TensorSpec
from mlflow.utils.os import is_windows
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout
def infer_or_get_default_signature(pipeline, example=None, model_config=None, flavor_config=None) -> ModelSignature:
    """
    Assigns a default ModelSignature for a given Pipeline type that has pyfunc support. These
    default signatures should only be generated and assigned when saving a model iff the user
    has not supplied a signature.
    For signature inference in some Pipelines that support complex input types, an input example
    is needed.
    """
    if example:
        try:
            timeout = MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT.get()
            if timeout and is_windows():
                timeout = None
                _logger.warning('On Windows, timeout is not supported for model signature inference. Therefore, the operation is not bound by a timeout and may hang indefinitely. If it hangs, please consider specifying the signature manually.')
            return _infer_signature_with_example(pipeline, example, model_config, flavor_config, timeout)
        except Exception as e:
            if isinstance(e, MlflowTimeoutError):
                msg = f'Attempted to generate a signature for the saved pipeline but prediction timed out after {timeout} seconds. Falling back to the default signature for the pipeline. You can specify a signature manually or increase the timeout by setting the environment variable {MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT}'
            else:
                msg = f'Attempted to generate a signature for the saved pipeline but encountered an error. Fall back to the default signature for the pipeline type. Error: {e}'
            _logger.warning(msg)
    import transformers
    for pipeline_type, signature in _DEFAULT_SIGNATURE_FOR_PIPELINES.items():
        if isinstance(pipeline, getattr(transformers, pipeline_type)):
            return signature
    _logger.warning('An unsupported Pipeline type was supplied for signature inference. Either provide an `input_example` or generate a signature manually via `infer_signature` to have a signature recorded in the MLmodel file.')