import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_lb_reload_static_resolve(self):
    t = template_format.parse(inline_templates.as_template)
    properties = t['Resources']['ElasticLoadBalancer']['Properties']
    properties['AvailabilityZones'] = {'Fn::GetAZs': ''}
    self.patchobject(stk_defn.StackDefinition, 'get_availability_zones', return_value=['abc', 'xyz'])
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    lb = stack['ElasticLoadBalancer']
    lb.state_set(lb.CREATE, lb.COMPLETE)
    lb.handle_update = mock.Mock(return_value=None)
    group = stack['WebServerGroup']
    self.setup_mocks(group, ['aaaabbbbcccc'])
    group._lb_reload()
    lb.handle_update.assert_called_once_with(mock.ANY, mock.ANY, {'Instances': ['aaaabbbbcccc']})