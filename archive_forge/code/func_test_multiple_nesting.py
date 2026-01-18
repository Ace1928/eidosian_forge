from unittest import mock
from urllib import parse
import fixtures
import sqlalchemy
from sqlalchemy import Boolean, Index, Integer, DateTime, String
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import ForeignKey, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import psycopg2
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import column_property
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import sql
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql import select
from sqlalchemy.types import UserDefinedType
from oslo_db import exception
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_multiple_nesting(self):
    callable_fn = mock.Mock(default=mock.Mock(return_value=None), mysql=mock.Mock(return_value=None))
    dispatcher = utils.dispatch_for_dialect('*', multiple=True)(callable_fn.default)
    dispatcher = dispatcher.dispatch_for('mysql+mysqlconnector')(dispatcher.dispatch_for('mysql+mysqldb')(callable_fn.mysql))
    mysqldb_url = utils.make_url('mysql+mysqldb://')
    mysqlconnector_url = utils.make_url('mysql+mysqlconnector://')
    sqlite_url = utils.make_url('sqlite://')
    dispatcher(mysqldb_url, 1)
    dispatcher(mysqlconnector_url, 2)
    dispatcher(sqlite_url, 3)
    self.assertEqual([mock.call.mysql(mysqldb_url, 1), mock.call.default(mysqldb_url, 1), mock.call.mysql(mysqlconnector_url, 2), mock.call.default(mysqlconnector_url, 2), mock.call.default(sqlite_url, 3)], callable_fn.mock_calls)