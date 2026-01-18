import time
from ... import exc
from ... import inspect
from ... import text
from ...testing import warn_test_suite
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_post_tables
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import prepare_for_drop_tables
from ...testing.provision import set_default_schema_on_connection
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert
@drop_all_schema_objects_post_tables.for_db('postgresql')
def drop_all_schema_objects_post_tables(cfg, eng):
    from sqlalchemy.dialects import postgresql
    inspector = inspect(eng)
    with eng.begin() as conn:
        for enum in inspector.get_enums('*'):
            conn.execute(postgresql.DropEnumType(postgresql.ENUM(name=enum['name'], schema=enum['schema'])))