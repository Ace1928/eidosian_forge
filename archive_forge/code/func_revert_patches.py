import abc
import functools
import inspect
import itertools
import typing
import uuid
from abc import abstractmethod
from contextlib import contextmanager
import mlflow
import mlflow.utils.autologging_utils
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import _MLFLOW_AUTOLOGGING_TESTING
from mlflow.tracking.client import MlflowClient
from mlflow.utils import gorilla, is_iterator
from mlflow.utils.autologging_utils import _logger
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING
def revert_patches(autologging_integration):
    """Reverts all patches on the specified destination class for autologging disablement purposes.

    Args:
        autologging_integration: The name of the autologging integration associated with the
            patch. Note: If called via fluent api (`autologging_integration="mlflow"`), then revert
            all patches for all active autologging integrations.

    """
    for patch in _AUTOLOGGING_PATCHES.get(autologging_integration, []):
        gorilla.revert(patch)
    _AUTOLOGGING_PATCHES.pop(autologging_integration, None)