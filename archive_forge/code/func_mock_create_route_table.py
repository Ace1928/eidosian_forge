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
def mock_create_route_table(self):
    self.rt_name = utils.PhysName('test_stack', 'the_route_table')
    self.mockclient.create_router.return_value = {'router': {'status': 'BUILD', 'name': self.rt_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'ffff'}}
    show_router_returns = [{'router': {'status': 'BUILD', 'name': self.rt_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'ffff'}}]
    for i in range(3):
        show_router_returns.append({'router': {'status': 'ACTIVE', 'name': self.rt_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'ffff'}})
    self.mockclient.show_router.side_effect = show_router_returns