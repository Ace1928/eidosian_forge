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
def test_compare_obj_with_unset_in_db_dict_ignored(self):
    my_obj = self.MyComparedObject(foo=1, bar=2)
    my_db_obj = {'foo': 1}
    ignore = ['bar']
    fixture.compare_obj(self, my_obj, my_db_obj, allow_missing=ignore)