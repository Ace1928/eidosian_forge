from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_consume_in_threads(self):
    self.conn.servers = [mock.Mock(), mock.Mock()]
    servs = self.conn.consume_in_threads()
    for serv in self.conn.servers:
        serv.start.assert_called_once_with()
    self.assertEqual(servs, self.conn.servers)