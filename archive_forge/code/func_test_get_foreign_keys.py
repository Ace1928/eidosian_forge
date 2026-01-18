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
@testing.combinations((False,), (True, testing.requires.schemas), argnames='use_schema')
@testing.requires.foreign_key_constraint_reflection
def test_get_foreign_keys(self, connection, use_schema):
    if use_schema:
        schema = config.test_schema
    else:
        schema = None
    users, addresses = (self.tables.users, self.tables.email_addresses)
    insp = inspect(connection)
    expected_schema = schema
    if testing.requires.self_referential_foreign_keys.enabled:
        users_fkeys = insp.get_foreign_keys(users.name, schema=schema)
        fkey1 = users_fkeys[0]
        with testing.requires.named_constraints.fail_if():
            eq_(fkey1['name'], 'user_id_fk')
        eq_(fkey1['referred_schema'], expected_schema)
        eq_(fkey1['referred_table'], users.name)
        eq_(fkey1['referred_columns'], ['user_id'])
        eq_(fkey1['constrained_columns'], ['parent_user_id'])
    addr_fkeys = insp.get_foreign_keys(addresses.name, schema=schema)
    fkey1 = addr_fkeys[0]
    with testing.requires.implicitly_named_constraints.fail_if():
        is_true(fkey1['name'] is not None)
    eq_(fkey1['referred_schema'], expected_schema)
    eq_(fkey1['referred_table'], users.name)
    eq_(fkey1['referred_columns'], ['user_id'])
    eq_(fkey1['constrained_columns'], ['remote_user_id'])
    no_cst = self.tables.no_constraints.name
    eq_(insp.get_foreign_keys(no_cst, schema=schema), [])