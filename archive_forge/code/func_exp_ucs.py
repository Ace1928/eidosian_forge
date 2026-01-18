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
def exp_ucs(self, schema=None, scope=ObjectScope.ANY, kind=ObjectKind.ANY, filter_names=None, all_=False):

    def uc(*cols, name, duplicates_index=None, is_index=False, comment=None):
        req = testing.requires.unique_index_reflect_as_unique_constraints
        if is_index and (not req.enabled):
            return ()
        res = {'column_names': list(cols), 'name': name, 'comment': comment}
        if duplicates_index:
            res['duplicates_index'] = duplicates_index
        return [res]
    materialized = {(schema, 'dingalings_v'): []}
    views = {(schema, 'email_addresses_v'): [], (schema, 'users_v'): [], (schema, 'user_tmp_v'): []}
    self._resolve_views(views, materialized)
    tables = {(schema, 'users'): [*uc('test1', 'test2', name='users_t_idx', duplicates_index='users_t_idx', is_index=True)], (schema, 'dingalings'): [*uc('data', name=mock.ANY, duplicates_index=mock.ANY), *uc('address_id', 'dingaling_id', name='zz_dingalings_multiple', duplicates_index='zz_dingalings_multiple', comment='di unique comment')], (schema, 'email_addresses'): [], (schema, 'comment_test'): [], (schema, 'no_constraints'): [], (schema, 'local_table'): [], (schema, 'remote_table'): [], (schema, 'remote_table_2'): [], (schema, 'noncol_idx_test_nopk'): [], (schema, 'noncol_idx_test_pk'): [], (schema, self.temp_table_name()): [*uc('name', name=f'user_tmp_uq_{config.ident}')]}
    if all_:
        return {**materialized, **views, **tables}
    else:
        res = self._resolve_kind(kind, tables, views, materialized)
        res = self._resolve_names(schema, scope, filter_names, res)
        return res