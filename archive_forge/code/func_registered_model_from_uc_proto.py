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
def registered_model_from_uc_proto(uc_proto: ProtoRegisteredModel) -> RegisteredModel:
    return RegisteredModel(name=uc_proto.name, creation_timestamp=uc_proto.creation_timestamp, last_updated_timestamp=uc_proto.last_updated_timestamp, description=uc_proto.description, aliases=[RegisteredModelAlias(alias=alias.alias, version=alias.version) for alias in uc_proto.aliases or []], tags=[RegisteredModelTag(key=tag.key, value=tag.value) for tag in uc_proto.tags or []])