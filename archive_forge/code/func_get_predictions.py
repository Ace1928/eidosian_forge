import contextlib
import json
from mlflow.deployments.base import BaseDeploymentClient
from mlflow.deployments.databricks import DatabricksDeploymentClient, DatabricksEndpoint
from mlflow.deployments.interface import get_deploy_client, run_local
from mlflow.deployments.openai import OpenAIDeploymentClient
from mlflow.deployments.utils import get_deployments_target, set_deployments_target
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def get_predictions(self, predictions_format='dataframe', dtype=None):
    """Get the predictions returned from the MLflow Model Server in the specified format.

        Args:
            predictions_format: The format in which to return the predictions. Either
                ``"dataframe"`` or ``"ndarray"``.
            dtype: The NumPy datatype to which to coerce the predictions. Only used when
                the "ndarray" predictions_format is specified.

        Raises:
            Exception: If the predictions cannot be represented in the specified format.

        Returns:
            The predictions, represented in the specified format.

        """
    import numpy as np
    import pandas as pd
    from pandas.core.dtypes.common import is_list_like
    if predictions_format == 'dataframe':
        predictions = self['predictions']
        if isinstance(predictions, str):
            return pd.DataFrame(data=[predictions])
        if isinstance(predictions, dict) and (not any((is_list_like(p) and getattr(p, 'ndim', 1) == 1 for p in predictions.values()))):
            return pd.DataFrame(data=predictions, index=[0])
        return pd.DataFrame(data=predictions)
    elif predictions_format == 'ndarray':
        return np.array(self['predictions'], dtype)
    else:
        raise MlflowException(f"Unrecognized predictions format: '{predictions_format}'", INVALID_PARAMETER_VALUE)