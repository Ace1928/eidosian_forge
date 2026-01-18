from collections import abc
import datetime
from unittest import mock
from sqlalchemy import Column
from sqlalchemy import Integer, String
from sqlalchemy import event
from sqlalchemy.orm import declarative_base
from oslo_db.sqlalchemy import models
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_modelbase_several_iters(self):
    mb = ExtraKeysModel()
    it1 = iter(mb)
    it2 = iter(mb)
    self.assertFalse(it1 is it2)
    self.assertEqual(dict(mb), dict(it1))
    self.assertEqual(dict(mb), dict(it2))