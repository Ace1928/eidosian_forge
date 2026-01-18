from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import RunInfo as ProtoRunInfo
def check_run_is_active(run_info):
    if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
        raise MlflowException(f"The run {run_info.run_id} must be in 'active' lifecycle_stage.", error_code=INVALID_PARAMETER_VALUE)