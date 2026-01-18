from typing import List, Optional
from sqlalchemy.exc import IntegrityError, MultipleResultsFound, NoResultFound
from sqlalchemy.orm import sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.server.auth.db import utils as dbutils
from mlflow.server.auth.db.models import (
from mlflow.server.auth.entities import ExperimentPermission, RegisteredModelPermission, User
from mlflow.server.auth.permissions import _validate_permission
from mlflow.store.db.utils import _get_managed_session_maker, create_sqlalchemy_engine_with_retry
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import _validate_username
def list_registered_model_permissions(self, username: str) -> List[RegisteredModelPermission]:
    with self.ManagedSessionMaker() as session:
        user = self._get_user(session, username=username)
        perms = session.query(SqlRegisteredModelPermission).filter(SqlRegisteredModelPermission.user_id == user.id).all()
        return [p.to_mlflow_entity() for p in perms]