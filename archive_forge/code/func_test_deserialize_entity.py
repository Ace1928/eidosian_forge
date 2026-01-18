from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_deserialize_entity(self):
    self.mock_base.deserialize_entity.return_value = 'foo'
    deser_ent = self.ser.deserialize_entity('context', 'entity')
    self.mock_base.deserialize_entity.assert_called_once_with('context', 'entity')
    self.assertEqual('foo', deser_ent)