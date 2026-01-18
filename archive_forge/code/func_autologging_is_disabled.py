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
def autologging_is_disabled(integration_name):
    """Returns a boolean flag of whether the autologging integration is disabled.

    Args:
        integration_name: An autologging integration flavor name.

    """
    explicit_disabled = get_autologging_config(integration_name, 'disable', True)
    if explicit_disabled:
        return True
    if integration_name in FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY and (not is_flavor_supported_for_associated_package_versions(integration_name)):
        return get_autologging_config(integration_name, 'disable_for_unsupported_versions', False)
    return False