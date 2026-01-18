import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_hydration(self):
    primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.5', 'versioned_object.data': {'foo': 1}}
    real_method = MyObj._obj_from_primitive

    def _obj_from_primitive(*args):
        return real_method(*args)
    with mock.patch.object(MyObj, '_obj_from_primitive') as ofp:
        ofp.side_effect = _obj_from_primitive
        obj = MyObj.obj_from_primitive(primitive)
        ofp.assert_called_once_with(None, '1.5', primitive)
    self.assertEqual(obj.foo, 1)