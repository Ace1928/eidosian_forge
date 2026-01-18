import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
def test_bad_initial_value(self):

    @obj_base.VersionedObjectRegistry.register
    class AnObject(obj_base.VersionedObject):
        fields = {'status': FakeStateMachineField()}
    obj = AnObject()
    with testtools.ExpectedException(ValueError):
        obj.status = 'FOO'