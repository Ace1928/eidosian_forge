import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_compare_obj_with_subs(self):
    mock_test = mock.Mock()
    mock_test.assertEqual = mock.Mock()
    my_obj = self.MyComparedObject(foo=1, bar=2)
    my_db_obj = {'doo': 1, 'bar': 2}
    subs = {'foo': 'doo'}
    fixture.compare_obj(mock_test, my_obj, my_db_obj, subs=subs)
    expected_calls = [(1, 1), (2, 2)]
    actual_calls = [c[0] for c in mock_test.assertEqual.call_args_list]
    for call in expected_calls:
        self.assertIn(call, actual_calls)