from __future__ import annotations
import sqlalchemy as sa
from .. import assertions
from .. import config
from ..assertions import eq_
from ..util import drop_all_tables_from_metadata
from ... import Column
from ... import func
from ... import Integer
from ... import select
from ... import Table
from ...orm import DeclarativeBase
from ...orm import MappedAsDataclass
from ...orm import registry
@config.fixture()
def async_testing_engine(self, testing_engine):

    def go(**kw):
        kw['asyncio'] = True
        return testing_engine(**kw)
    return go