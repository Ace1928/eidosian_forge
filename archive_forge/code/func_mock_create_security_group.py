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
def mock_create_security_group(self):
    self.sg_name = utils.PhysName('test_stack', 'the_sg')
    self.mockclient.create_security_group.return_value = {'security_group': {'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'name': self.sg_name, 'description': 'SSH access', 'security_group_rules': [], 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}}
    self.create_security_group_rule_expected = {'security_group_rule': {'direction': 'ingress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 22, 'ethertype': 'IPv4', 'port_range_max': 22, 'protocol': 'tcp', 'security_group_id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}}
    self.mockclient.create_security_group_rule.return_value = {'security_group_rule': {'direction': 'ingress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 22, 'ethertype': 'IPv4', 'port_range_max': 22, 'protocol': 'tcp', 'security_group_id': '0389f747-7785-4757-b7bb-2ab07e4b09c3', 'id': 'bbbb'}}