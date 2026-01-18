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
@prepare_for_drop_tables.for_db('postgresql')
def prepare_for_drop_tables(config, connection):
    """Ensure there are no locks on the current username/database."""
    result = connection.exec_driver_sql("select pid, state, wait_event_type, query from pg_stat_activity where usename=current_user and datname=current_database() and state='idle in transaction' and pid != pg_backend_pid()")
    rows = result.all()
    if rows:
        warn_test_suite('PostgreSQL may not be able to DROP tables due to idle in transaction: %s' % '; '.join((row._mapping['query'] for row in rows)))