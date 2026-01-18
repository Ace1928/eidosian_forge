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
def get_metric_history_bulk(self, run_ids, metric_key, max_results):
    """
        Return all logged values for a given metric.

        Args:
            run_ids: Unique identifiers of the runs from which to fetch the metric histories for
                the specified key.
            metric_key: Metric name within the runs.
            max_results: The maximum number of results to return.

        Returns:
            A List of SqlAlchemyStore.MetricWithRunId objects if metric_key values have been logged
            to one or more of the specified run_ids, else an empty list. Results are sorted by run
            ID in lexicographically ascending order, followed by timestamp, step, and value in
            numerically ascending order.
        """
    with self.ManagedSessionMaker() as session:
        metrics = session.query(SqlMetric).filter(SqlMetric.key == metric_key, SqlMetric.run_uuid.in_(run_ids)).order_by(SqlMetric.run_uuid, SqlMetric.timestamp, SqlMetric.step, SqlMetric.value).limit(max_results).all()
        return [MetricWithRunId(run_id=metric.run_uuid, metric=metric.to_mlflow_entity()) for metric in metrics]