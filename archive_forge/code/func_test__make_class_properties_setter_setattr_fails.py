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
@mock.patch.object(obj_base.LOG, 'exception')
def test__make_class_properties_setter_setattr_fails(self, mock_log):

    @obj_base.VersionedObjectRegistry.register
    class AnObject(obj_base.VersionedObject):
        fields = {'intfield': fields.IntegerField()}
    with mock.patch.object(obj_base, '_get_attrname') as mock_attr:
        mock_attr.return_value = '__class__'
        self.assertRaises(TypeError, AnObject, intfield=2)
        mock_attr.assert_called_once_with('intfield')
        mock_log.assert_called_once_with(mock.ANY, {'attr': 'AnObject.intfield'})