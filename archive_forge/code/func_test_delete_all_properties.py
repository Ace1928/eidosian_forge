from unittest import mock
from keystoneauth1 import adapter
from openstack import exceptions
from openstack.image.v2 import metadef_namespace
from openstack.tests.unit import base
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
def test_delete_all_properties(self):
    sot = metadef_namespace.MetadefNamespace(**EXAMPLE)
    session = mock.Mock(spec=adapter.Adapter)
    sot._translate_response = mock.Mock()
    sot.delete_all_properties(session)
    session.delete.assert_called_with('metadefs/namespaces/OS::Cinder::Volumetype/properties')