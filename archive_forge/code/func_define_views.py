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
@classmethod
def define_views(cls, metadata, schema):
    if testing.requires.materialized_views.enabled:
        materialized = {'dingalings'}
    else:
        materialized = set()
    for table_name in ('users', 'email_addresses', 'dingalings'):
        fullname = table_name
        if schema:
            fullname = f'{schema}.{table_name}'
        view_name = fullname + '_v'
        prefix = 'MATERIALIZED ' if table_name in materialized else ''
        query = f'CREATE {prefix}VIEW {view_name} AS SELECT * FROM {fullname}'
        event.listen(metadata, 'after_create', DDL(query))
        if table_name in materialized:
            index_name = 'mat_index'
            if schema and testing.against('oracle'):
                index_name = f'{schema}.{index_name}'
            idx = f'CREATE INDEX {index_name} ON {view_name}(data)'
            event.listen(metadata, 'after_create', DDL(idx))
        event.listen(metadata, 'before_drop', DDL(f'DROP {prefix}VIEW {view_name}'))