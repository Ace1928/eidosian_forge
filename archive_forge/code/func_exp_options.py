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
def exp_options(self, schema=None, scope=ObjectScope.ANY, kind=ObjectKind.ANY, filter_names=None):
    materialized = {(schema, 'dingalings_v'): mock.ANY}
    views = {(schema, 'email_addresses_v'): mock.ANY, (schema, 'users_v'): mock.ANY, (schema, 'user_tmp_v'): mock.ANY}
    self._resolve_views(views, materialized)
    tables = {(schema, 'users'): mock.ANY, (schema, 'dingalings'): mock.ANY, (schema, 'email_addresses'): mock.ANY, (schema, 'comment_test'): mock.ANY, (schema, 'no_constraints'): mock.ANY, (schema, 'local_table'): mock.ANY, (schema, 'remote_table'): mock.ANY, (schema, 'remote_table_2'): mock.ANY, (schema, 'noncol_idx_test_nopk'): mock.ANY, (schema, 'noncol_idx_test_pk'): mock.ANY, (schema, self.temp_table_name()): mock.ANY}
    res = self._resolve_kind(kind, tables, views, materialized)
    res = self._resolve_names(schema, scope, filter_names, res)
    return res