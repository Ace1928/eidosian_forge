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
def test_instance_attributes(self):
    fake_instance = FakeDBInstance()
    self.client.instances.create.return_value = fake_instance
    self.client.instances.get.return_value = fake_instance
    t = template_format.parse(db_template)
    instance = self._setup_test_instance('attr_test', t)
    self.assertEqual('testhost', instance.FnGetAtt('hostname'))
    self.assertEqual('https://adga23dd432a.rackspacecloud.com/132345245', instance.FnGetAtt('href'))