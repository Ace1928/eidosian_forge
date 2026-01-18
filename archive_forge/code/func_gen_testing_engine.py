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
def gen_testing_engine(url=None, options=None, future=None, asyncio=False, transfer_staticpool=False, share_pool=False):
    if options is None:
        options = {}
    options['scope'] = 'fixture'
    return engines.testing_engine(url=url, options=options, asyncio=asyncio, transfer_staticpool=transfer_staticpool, share_pool=share_pool)