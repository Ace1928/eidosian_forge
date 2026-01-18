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
@testing.combinations((True, testing.requires.schemas), (False,), argnames='use_schema')
def test_get_table_options(self, use_schema):
    insp = inspect(config.db)
    schema = config.test_schema if use_schema else None
    if testing.requires.reflect_table_options.enabled:
        res = insp.get_table_options('users', schema=schema)
        is_true(isinstance(res, dict))
        res = insp.get_table_options('no_constraints', schema=schema)
        is_true(isinstance(res, dict))
    else:
        with expect_raises(NotImplementedError):
            res = insp.get_table_options('users', schema=schema)