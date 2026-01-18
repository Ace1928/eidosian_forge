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
def test_handle_update_databases(self, mock_client, mock_plugin):
    prop_diff = {'databases': [{'name': 'bar', 'character_set': 'ascii'}, {'name': 'baz'}]}
    mget = mock_client().databases.list
    mbar = mock.Mock(name='bar')
    mbar.name = 'bar'
    mbiff = mock.Mock(name='biff')
    mbiff.name = 'biff'
    mget.return_value = [mbar, mbiff]
    trove = dbinstance.Instance('test', self._rdef, self._stack)
    expected = {'databases': [{'character_set': 'ascii', 'name': 'bar'}, {'ACTION': 'CREATE', 'name': 'baz'}, {'ACTION': 'DELETE', 'name': 'biff'}]}
    self.assertEqual(expected, trove.handle_update(None, None, prop_diff))