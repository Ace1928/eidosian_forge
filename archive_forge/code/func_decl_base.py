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
@config.fixture
def decl_base(self, metadata):
    _md = metadata

    class Base(DeclarativeBase):
        metadata = _md
        type_annotation_map = {str: sa.String().with_variant(sa.String(50), 'mysql', 'mariadb', 'oracle')}
    yield Base
    Base.registry.dispose()