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
@testing.requires.primary_key_constraint_reflection
def test_get_pk_constraint(self, connection, use_schema):
    if use_schema:
        schema = testing.config.test_schema
    else:
        schema = None
    users, addresses = (self.tables.users, self.tables.email_addresses)
    insp = inspect(connection)
    exp = self.exp_pks(schema=schema)
    users_cons = insp.get_pk_constraint(users.name, schema=schema)
    self._check_list([users_cons], [exp[schema, users.name]], self._required_pk_keys)
    addr_cons = insp.get_pk_constraint(addresses.name, schema=schema)
    exp_cols = exp[schema, addresses.name]['constrained_columns']
    eq_(addr_cons['constrained_columns'], exp_cols)
    with testing.requires.reflects_pk_names.fail_if():
        eq_(addr_cons['name'], 'email_ad_pk')
    no_cst = self.tables.no_constraints.name
    self._check_list([insp.get_pk_constraint(no_cst, schema=schema)], [exp[schema, no_cst]], self._required_pk_keys)