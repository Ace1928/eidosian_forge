import functools
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def setup_keystone_mocks_with_mock(test_case, stack):
    fkc = fake_ks.FakeKeystoneClient()
    test_case.patchobject(keystone.KeystoneClientPlugin, '_create')
    keystone.KeystoneClientPlugin._create.return_value = fkc