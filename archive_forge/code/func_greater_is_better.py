from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import deprecated
@property
def greater_is_better(self):
    """
        Boolean value representing whether higher value is better for the metric.
        """
    return self._greater_is_better