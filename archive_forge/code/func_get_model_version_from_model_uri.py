import datetime as dt
import decimal
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.store.artifact.utils.models import get_model_name_and_version
from mlflow.types import DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.types.schema import Array, Map, Object, Property
from mlflow.types.utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import (
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
def get_model_version_from_model_uri(model_uri):
    """
    Helper function to fetch a model version from a model uri of the form
    models:/<model_name>/<model_version/stage/latest>.
    """
    import mlflow
    from mlflow import MlflowClient
    databricks_profile_uri = get_databricks_profile_uri_from_artifact_uri(model_uri) or mlflow.get_registry_uri()
    client = MlflowClient(registry_uri=databricks_profile_uri)
    name, version = get_model_name_and_version(client, model_uri)
    return client.get_model_version(name, version)