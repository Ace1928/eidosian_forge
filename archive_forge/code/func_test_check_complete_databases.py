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
def test_check_complete_databases(self, mock_client, mock_plugin):
    mock_instance = mock.Mock(status='ACTIVE', name='mock_instance')
    mock_client().instances.get.return_value = mock_instance
    updates = {'databases': [{'name': 'bar', 'character_set': 'ascii'}, {'ACTION': 'CREATE', 'name': 'baz'}, {'ACTION': 'DELETE', 'name': 'biff'}]}
    trove = dbinstance.Instance('test', self._rdef, self._stack)
    self.assertTrue(trove.check_update_complete(updates))
    mcreate = mock_client().databases.create
    mdelete = mock_client().databases.delete
    mcreate.assert_called_once_with(mock_instance, [{'name': 'baz'}])
    mdelete.assert_called_once_with(mock_instance, 'biff')