from __future__ import annotations
from typing import Any
import sqlalchemy as sa
from .base import TestBase
from .sql import TablesTest
from .. import assertions
from .. import config
from .. import schema
from ..entities import BasicEntity
from ..entities import ComparableEntity
from ..util import adict
from ... import orm
from ...orm import DeclarativeBase
from ...orm import events as orm_events
from ...orm import registry
@config.fixture
def fixture_session(self):
    return fixture_session()