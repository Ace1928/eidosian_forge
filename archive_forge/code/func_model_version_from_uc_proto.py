from typing import List, Optional
from mlflow.entities.model_registry import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import TemporaryCredentials
from mlflow.store.artifact.artifact_repo import ArtifactRepository
def model_version_from_uc_proto(uc_proto: ProtoModelVersion) -> ModelVersion:
    return ModelVersion(name=uc_proto.name, version=uc_proto.version, creation_timestamp=uc_proto.creation_timestamp, last_updated_timestamp=uc_proto.last_updated_timestamp, description=uc_proto.description, user_id=uc_proto.user_id, source=uc_proto.source, run_id=uc_proto.run_id, status=uc_model_version_status_to_string(uc_proto.status), status_message=uc_proto.status_message, aliases=[alias.alias for alias in uc_proto.aliases or []], tags=[ModelVersionTag(key=tag.key, value=tag.value) for tag in uc_proto.tags or []])