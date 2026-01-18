import abc
import functools
import logging
import pprint
import re
import alembic
import alembic.autogenerate
import alembic.migration
import sqlalchemy
import sqlalchemy.exc
import sqlalchemy.sql.expression as expr
import sqlalchemy.types as types
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def test_models_sync(self):
    engine = self.get_engine()
    backend = provision.Backend(engine.name, engine.url)
    self.addCleanup(functools.partial(backend.drop_all_objects, engine))
    self.db_sync(self.get_engine())
    with self.get_engine().connect() as conn:
        opts = {'include_object': self.include_object, 'compare_type': self.compare_type, 'compare_server_default': self.compare_server_default}
        mc = alembic.migration.MigrationContext.configure(conn, opts=opts)
        diff = self.filter_metadata_diff(alembic.autogenerate.compare_metadata(mc, self.get_metadata()))
        if diff:
            msg = pprint.pformat(diff, indent=2, width=20)
            self.fail("Models and migration scripts aren't in sync:\n%s" % msg)