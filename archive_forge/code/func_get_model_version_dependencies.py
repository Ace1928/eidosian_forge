import base64
import functools
import logging
import os
import shutil
from contextlib import contextmanager
import mlflow
from mlflow.entities import Run
from mlflow.environment_variables import MLFLOW_UNITY_CATALOG_PRESIGNED_URLS_ENABLED
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.protos.service_pb2 import GetRun, MlflowService
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds, is_databricks_uri
from mlflow.utils.mlflow_tags import (
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import (
def get_model_version_dependencies(model_dir):
    """
    Gets the specified dependencies for a particular model version and formats them
    to be passed into CreateModelVersion.
    """
    from mlflow.langchain.databricks_dependencies import _DATABRICKS_CHAT_ENDPOINT_NAME_KEY, _DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY, _DATABRICKS_LLM_ENDPOINT_NAME_KEY, _DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY
    model = _load_model(model_dir)
    model_info = model.get_model_info()
    dependencies = []
    index_names = _fetch_langchain_dependency_from_model_info(model_info, _DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY)
    for index_name in index_names:
        dependencies.append({'type': 'DATABRICKS_VECTOR_INDEX', 'name': index_name})
    for key in (_DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY, _DATABRICKS_LLM_ENDPOINT_NAME_KEY, _DATABRICKS_CHAT_ENDPOINT_NAME_KEY):
        endpoint_names = _fetch_langchain_dependency_from_model_info(model_info, key)
        for endpoint_name in endpoint_names:
            dependencies.append({'type': 'DATABRICKS_MODEL_ENDPOINT', 'name': endpoint_name})
    return dependencies