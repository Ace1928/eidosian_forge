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
@kind
def test_has_index(self, kind, connection, metadata):
    meth = self._has_index(kind, connection)
    assert meth('test_table', 'my_idx')
    assert not meth('test_table', 'my_idx_s')
    assert not meth('nonexistent_table', 'my_idx')
    assert not meth('test_table', 'nonexistent_idx')
    assert not meth('test_table', 'my_idx_2')
    assert not meth('test_table_2', 'my_idx_3')
    idx = Index('my_idx_2', self.tables.test_table.c.data2)
    tbl = Table('test_table_2', metadata, Column('foo', Integer), Index('my_idx_3', 'foo'))
    idx.create(connection)
    tbl.create(connection)
    try:
        if kind == 'inspector':
            assert not meth('test_table', 'my_idx_2')
            assert not meth('test_table_2', 'my_idx_3')
            meth.__self__.clear_cache()
        assert meth('test_table', 'my_idx_2') is True
        assert meth('test_table_2', 'my_idx_3') is True
    finally:
        tbl.drop(connection)
        idx.drop(connection)