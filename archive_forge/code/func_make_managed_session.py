import logging
import os
import time
from contextlib import contextmanager
import sqlalchemy
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import sql
from sqlalchemy.pool import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR, TEMPORARILY_UNAVAILABLE
from mlflow.store.db.db_types import SQLITE
from mlflow.store.model_registry.dbmodels.models import (
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.store.tracking.dbmodels.models import (
@contextmanager
def make_managed_session():
    """Provide a transactional scope around a series of operations."""
    with SessionMaker() as session:
        try:
            if db_type == SQLITE:
                session.execute(sql.text('PRAGMA foreign_keys = ON;'))
                session.execute(sql.text('PRAGMA busy_timeout = 20000;'))
                session.execute(sql.text('PRAGMA case_sensitive_like = true;'))
            yield session
            session.commit()
        except MlflowException:
            session.rollback()
            raise
        except sqlalchemy.exc.OperationalError as e:
            session.rollback()
            _logger.exception('SQLAlchemy database error. The following exception is caught.\n%s', e)
            raise MlflowException(message=e, error_code=TEMPORARILY_UNAVAILABLE)
        except sqlalchemy.exc.SQLAlchemyError as e:
            session.rollback()
            raise MlflowException(message=e, error_code=BAD_REQUEST)
        except Exception as e:
            session.rollback()
            raise MlflowException(message=e, error_code=INTERNAL_ERROR)