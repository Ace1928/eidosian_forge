from __future__ import annotations
import configparser
from contextlib import contextmanager
import io
import re
from typing import Any
from typing import Dict
from sqlalchemy import Column
from sqlalchemy import inspect
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import testing
from sqlalchemy import text
from sqlalchemy.testing import config
from sqlalchemy.testing import mock
from sqlalchemy.testing.assertions import eq_
from sqlalchemy.testing.fixtures import TablesTest as SQLAlchemyTablesTest
from sqlalchemy.testing.fixtures import TestBase as SQLAlchemyTestBase
import alembic
from .assertions import _get_dialect
from ..environment import EnvironmentContext
from ..migration import MigrationContext
from ..operations import Operations
from ..util import sqla_compat
from ..util.sqla_compat import create_mock_engine
from ..util.sqla_compat import sqla_14
from ..util.sqla_compat import sqla_2
def clear_assertions(self):
    buf.lines[:] = []