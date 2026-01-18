import contextlib
import inspect
import logging
import time
from typing import List
import mlflow
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient  # noqa: F401
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.autologging_utils.safety import (  # noqa: F401
from mlflow.utils.autologging_utils.versioning import (
def resolve_input_example_and_signature(get_input_example, infer_model_signature, log_input_example, log_model_signature, logger):
    """Handles the logic of calling functions to gather the input example and infer the model
    signature.

    Args:
        get_input_example: Function which returns an input example, usually sliced from a
            dataset. This function can raise an exception, its message will be
            shown to the user in a warning in the logs.
        infer_model_signature: Function which takes an input example and returns the signature
            of the inputs and outputs of the model. This function can raise
            an exception, its message will be shown to the user in a warning
            in the logs.
        log_input_example: Whether to log errors while collecting the input example, and if it
            succeeds, whether to return the input example to the user. We collect
            it even if this parameter is False because it is needed for inferring
            the model signature.
        log_model_signature: Whether to infer and return the model signature.
        logger: The logger instance used to log warnings to the user during input example
            collection and model signature inference.

    Returns:
        A tuple of input_example and signature. Either or both could be None based on the
        values of log_input_example and log_model_signature.

    """
    input_example = None
    input_example_user_msg = None
    input_example_failure_msg = None
    if log_input_example or log_model_signature:
        try:
            input_example = get_input_example()
        except Exception as e:
            input_example_failure_msg = str(e)
            input_example_user_msg = 'Failed to gather input example: ' + str(e)
    model_signature = None
    model_signature_user_msg = None
    if log_model_signature:
        try:
            if input_example is None:
                raise Exception('could not sample data to infer model signature: ' + input_example_failure_msg)
            model_signature = infer_model_signature(input_example)
        except Exception as e:
            model_signature_user_msg = 'Failed to infer model signature: ' + str(e)
    if model_signature is None and input_example is not None and (not log_model_signature or model_signature_user_msg is not None):
        model_signature = False
    if log_input_example and input_example_user_msg is not None:
        logger.warning(input_example_user_msg)
    if log_model_signature and model_signature_user_msg is not None:
        logger.warning(model_signature_user_msg)
    return (input_example if log_input_example else None, model_signature)