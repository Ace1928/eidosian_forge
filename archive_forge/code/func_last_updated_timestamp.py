from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.protos.model_registry_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.model_registry_pb2 import ModelVersionTag as ProtoModelVersionTag
@last_updated_timestamp.setter
def last_updated_timestamp(self, updated_timestamp):
    self._last_updated_timestamp = updated_timestamp