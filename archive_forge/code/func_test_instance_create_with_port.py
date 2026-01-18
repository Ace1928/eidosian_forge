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
def test_instance_create_with_port(self):
    t = template_format.parse(db_template_with_nics)
    instance = self._setup_test_instance('dbinstance_test', t)
    self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', return_value='someportid')
    self.stub_PortConstraint_validate()
    scheduler.TaskRunner(instance.create)()
    self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
    self.client.instances.create.assert_called_once_with('test', '1', volume={'size': 30}, databases=[], users=[], restorePoint=None, availability_zone=None, datastore=None, datastore_version=None, nics=[{'port-id': 'someportid', 'v4-fixed-ip': '1.2.3.4'}], replica_of=None, replica_count=None)