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
@testing.requires.check_constraint_reflection
@_multi_combination
def test_get_multi_check_constraints(self, get_multi_exp, schema, scope, kind, use_filter):
    insp, kws, exp = get_multi_exp(schema, scope, kind, use_filter, Inspector.get_check_constraints, self.exp_ccs)
    for kw in kws:
        insp.clear_cache()
        result = insp.get_multi_check_constraints(**kw)
        self._adjust_sort(result, exp, lambda d: tuple(d['sqltext']))
        self._check_table_dict(result, exp, self._required_cc_keys)