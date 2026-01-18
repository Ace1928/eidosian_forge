import operator
import re
import sqlalchemy as sa
from .. import config
from .. import engines
from .. import eq_
from .. import expect_raises
from .. import expect_raises_message
from .. import expect_warnings
from .. import fixtures
from .. import is_
from ..provision import get_temp_table_name
from ..provision import temp_table_keyword_args
from ..schema import Column
from ..schema import Table
from ... import event
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import String
from ... import testing
from ... import types as sql_types
from ...engine import Inspector
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...exc import NoSuchTableError
from ...exc import UnreflectableTableError
from ...schema import DDL
from ...schema import Index
from ...sql.elements import quoted_name
from ...sql.schema import BLANK_SCHEMA
from ...testing import ComparesIndexes
from ...testing import ComparesTables
from ...testing import is_false
from ...testing import is_true
from ...testing import mock
@testing.requires.cross_schema_fk_reflection
@testing.requires.schemas
def test_get_inter_schema_foreign_keys(self, connection):
    local_table, remote_table, remote_table_2 = self.tables('%s.local_table' % connection.dialect.default_schema_name, '%s.remote_table' % testing.config.test_schema, '%s.remote_table_2' % testing.config.test_schema)
    insp = inspect(connection)
    local_fkeys = insp.get_foreign_keys(local_table.name)
    eq_(len(local_fkeys), 1)
    fkey1 = local_fkeys[0]
    eq_(fkey1['referred_schema'], testing.config.test_schema)
    eq_(fkey1['referred_table'], remote_table_2.name)
    eq_(fkey1['referred_columns'], ['id'])
    eq_(fkey1['constrained_columns'], ['remote_id'])
    remote_fkeys = insp.get_foreign_keys(remote_table.name, schema=testing.config.test_schema)
    eq_(len(remote_fkeys), 1)
    fkey2 = remote_fkeys[0]
    is_true(fkey2['referred_schema'] in (None, connection.dialect.default_schema_name))
    eq_(fkey2['referred_table'], local_table.name)
    eq_(fkey2['referred_columns'], ['id'])
    eq_(fkey2['constrained_columns'], ['local_id'])