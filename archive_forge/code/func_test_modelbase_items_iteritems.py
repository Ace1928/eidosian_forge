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
def test_modelbase_items_iteritems(self):
    h = {'a': '1', 'b': '2'}
    expected = {'id': None, 'smth': None, 'name': 'NAME', 'a': '1', 'b': '2'}
    self.ekm.update(h)
    self.assertEqual(expected, dict(self.ekm.items()))
    self.assertEqual(expected, dict(self.ekm.iteritems()))