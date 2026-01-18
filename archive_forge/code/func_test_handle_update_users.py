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
def test_handle_update_users(self, mock_client, mock_plugin):
    prop_diff = {'users': [{'name': 'baz', 'password': 'changed', 'databases': ['bar', 'biff']}, {'name': 'user2', 'password': 'password', 'databases': ['biff', 'bar']}]}
    uget = mock_client().users
    mbaz = mock.Mock(name='baz')
    mbaz.name = 'baz'
    mdel = mock.Mock(name='deleted')
    mdel.name = 'deleted'
    uget.list.return_value = [mbaz, mdel]
    trove = dbinstance.Instance('test', self._rdef, self._stack)
    expected = {'users': [{'databases': ['bar', 'biff'], 'name': 'baz', 'password': 'changed'}, {'ACTION': 'CREATE', 'databases': ['biff', 'bar'], 'name': 'user2', 'password': 'password'}, {'ACTION': 'DELETE', 'name': 'deleted'}]}
    self.assertEqual(expected, trove.handle_update(None, None, prop_diff))