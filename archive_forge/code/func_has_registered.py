import abc
import inspect
import entrypoints
from mlflow.deployments.base import BaseDeploymentClient
from mlflow.deployments.utils import parse_target_uri
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST
from mlflow.utils.annotations import developer_stable
@property
def has_registered(self):
    """
        Returns bool representing whether the "register_entrypoints" has run or not. This
        doesn't return True if `register` method is called outside of `register_entrypoints`
        to register plugins
        """
    return self._has_registered