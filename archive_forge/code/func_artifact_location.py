from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import ExperimentTag as ProtoExperimentTag
@property
def artifact_location(self):
    """String corresponding to the root artifact URI for the experiment."""
    return self._artifact_location