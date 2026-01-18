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
def clean_up_stack(test_case, stack, delete_res=True):
    if delete_res:
        fc = fakes_nova.FakeClient()
        test_case.patchobject(instances.Instance, 'client', return_value=fc)
        test_case.patchobject(fc.servers, 'delete', side_effect=fakes_nova.fake_exception())
    stack.delete()