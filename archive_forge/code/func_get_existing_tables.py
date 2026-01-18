from alembic import op
import sqlalchemy as sa
from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelAlias
def get_existing_tables():
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    return inspector.get_table_names()