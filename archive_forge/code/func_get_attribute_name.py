import sqlalchemy as sa
from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.mlflow_tags import _get_run_name_from_tags
from mlflow.utils.time import get_current_time_millis
@staticmethod
def get_attribute_name(mlflow_attribute_name):
    """
        Resolves an MLflow attribute name to a `SqlRun` attribute name.
        """
    return {'run_name': 'name', 'run_id': 'run_uuid'}.get(mlflow_attribute_name, mlflow_attribute_name)