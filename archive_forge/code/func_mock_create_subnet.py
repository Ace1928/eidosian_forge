from unittest import mock
import uuid
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.aws.ec2 import subnet as sn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def mock_create_subnet(self):
    self.subnet_name = utils.PhysName('test_stack', 'the_subnet')
    self.mockclient.create_subnet.return_value = {'subnet': {'status': 'ACTIVE', 'name': self.subnet_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'cccc'}}