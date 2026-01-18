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
def test_bad_transitions(self):

    @obj_base.VersionedObjectRegistry.register
    class AnObject(obj_base.VersionedObject):
        fields = {'status': FakeStateMachineField()}
    obj = AnObject(status='ERROR')
    try:
        obj.status = FakeStateMachineField.ACTIVE
    except ValueError as e:
        ex = e
    else:
        ex = None
    self.assertIsNotNone(ex, 'Invalid transition failed to raise error')
    self.assertEqual("AnObject.status is not allowed to transition out of 'ERROR' state to 'ACTIVE' state, choose from ['PENDING']", str(ex))