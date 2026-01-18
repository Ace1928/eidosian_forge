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
def test_good_transitions(self):

    @obj_base.VersionedObjectRegistry.register
    class AnObject(obj_base.VersionedObject):
        fields = {'status': FakeStateMachineField()}
    obj = AnObject()
    obj.status = FakeStateMachineField.ACTIVE
    obj.status = FakeStateMachineField.PENDING
    obj.status = FakeStateMachineField.ERROR
    obj.status = FakeStateMachineField.PENDING
    obj.status = FakeStateMachineField.ACTIVE