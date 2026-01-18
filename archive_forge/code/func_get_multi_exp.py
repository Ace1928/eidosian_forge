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
@testing.fixture
def get_multi_exp(self, connection):

    def provide_fixture(schema, scope, kind, use_filter, single_reflect_fn, exp_method):
        insp = inspect(connection)
        single_reflect_fn(insp, 'email_addresses')
        kw = {'scope': scope, 'kind': kind}
        if schema:
            schema = schema()
        filter_names = []
        if ObjectKind.TABLE in kind:
            filter_names.extend(['comment_test', 'users', 'does-not-exist'])
        if ObjectKind.VIEW in kind:
            filter_names.extend(['email_addresses_v', 'does-not-exist'])
        if ObjectKind.MATERIALIZED_VIEW in kind:
            filter_names.extend(['dingalings_v', 'does-not-exist'])
        if schema:
            kw['schema'] = schema
        if use_filter:
            kw['filter_names'] = filter_names
        exp = exp_method(schema=schema, scope=scope, kind=kind, filter_names=kw.get('filter_names'))
        kws = [kw]
        if scope == ObjectScope.DEFAULT:
            nkw = kw.copy()
            nkw.pop('scope')
            kws.append(nkw)
        if kind == ObjectKind.TABLE:
            nkw = kw.copy()
            nkw.pop('kind')
            kws.append(nkw)
        return (inspect(connection), kws, exp)
    return provide_fixture