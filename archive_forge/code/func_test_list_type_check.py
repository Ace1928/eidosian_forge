from sqlalchemy import Column, Integer
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import declarative_base
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import types
from oslo_db.tests.sqlalchemy import base as test_base
def test_list_type_check(self):
    self.assertRaises(db_exc.DBError, JsonTable(id=1, jlist={}).save, self.session)