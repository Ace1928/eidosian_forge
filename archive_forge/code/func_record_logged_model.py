import json
import logging
import math
import random
import threading
import time
import uuid
from functools import reduce
from typing import List, Optional
import sqlalchemy
import sqlalchemy.sql.expression as sql
from sqlalchemy import and_, func, sql, text
from sqlalchemy.future import select
import mlflow.store.db.utils
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.metric import MetricWithRunId
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.db.db_types import MSSQL, MYSQL
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.dbmodels.models import (
from mlflow.utils.file_utils import local_file_uri_to_path, mkdir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import (
from mlflow.utils.validation import (
def record_logged_model(self, run_id, mlflow_model):
    from mlflow.models import Model
    if not isinstance(mlflow_model, Model):
        raise TypeError(f"Argument 'mlflow_model' should be mlflow.models.Model, got '{type(mlflow_model)}'")
    model_dict = mlflow_model.to_dict()
    with self.ManagedSessionMaker() as session:
        run = self._get_run(run_uuid=run_id, session=session)
        self._check_run_is_active(run)
        previous_tag = [t for t in run.tags if t.key == MLFLOW_LOGGED_MODELS]
        if previous_tag:
            value = json.dumps(json.loads(previous_tag[0].value) + [model_dict])
        else:
            value = json.dumps([model_dict])
        _validate_tag(MLFLOW_LOGGED_MODELS, value)
        session.merge(SqlTag(key=MLFLOW_LOGGED_MODELS, value=value, run_uuid=run_id))