from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_check_client_exceptions(self, mock_client, mock_plugin):
    mock_instance = mock.Mock(status='ACTIVE')
    mock_client().instances.get.return_value = mock_instance
    mock_plugin().is_client_exception.return_value = True
    mock_plugin().is_over_limit.side_effect = [True, False]
    trove = dbinstance.Instance('test', self._rdef, self._stack)
    with mock.patch.object(trove, '_update_flavor') as mupdate:
        mupdate.side_effect = [Exception('test'), Exception("No change was requested because I'm testing")]
        self.assertFalse(trove.check_update_complete({'foo': 'bar'}))
        self.assertFalse(trove.check_update_complete({'foo': 'bar'}))
        self.assertEqual(2, mupdate.call_count)
        self.assertEqual(2, mock_plugin().is_client_exception.call_count)