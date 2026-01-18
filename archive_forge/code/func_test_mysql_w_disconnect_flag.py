import contextlib
import itertools
from unittest import mock
import sqlalchemy as sqla
from sqlalchemy import event
import sqlalchemy.exc
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy import sql
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db.tests import utils as test_utils
def test_mysql_w_disconnect_flag(self):
    for code in [2002, 2003, 2002]:
        self._test_ping_listener_disconnected('mysql', self.OperationalError('%d MySQL server has gone away' % code))