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
def compare_server_default(self, ctxt, ins_col, meta_col, insp_def, meta_def, rendered_meta_def):
    """Compare default values between model and db table.

        Return True if the defaults are different, False if not, or None to
        allow the default implementation to compare these defaults.

        :param ctxt: alembic MigrationContext instance
        :param insp_col: reflected column
        :param meta_col: column from model
        :param insp_def: reflected column default value
        :param meta_def: column default value from model
        :param rendered_meta_def: rendered column default value (from model)

        """
    return self._compare_server_default(ctxt.bind, meta_col, insp_def, meta_def)