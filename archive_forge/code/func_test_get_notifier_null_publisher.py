from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_get_notifier_null_publisher(self):
    mock_notifier = mock.Mock(return_value=None)
    messaging.Notifier.__init__ = mock_notifier
    rpc.get_notifier('service', host='bar')
    mock_notifier.assert_called_once_with(mock.ANY, serializer=mock.ANY, publisher_id='service.bar')