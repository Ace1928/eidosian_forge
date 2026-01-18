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
def test_attributes_format_fallback(self):
    self.instance_group.get_output = mock.Mock(return_value=['2.1.3.2', '2.1.3.1', '2.1.3.3'])
    mock_members = self.patchobject(grouputils, 'get_members')
    instances = []
    for ip_ex in range(1, 4):
        inst = mock.Mock()
        inst.FnGetAtt.return_value = '2.1.3.%d' % ip_ex
        instances.append(inst)
    mock_members.return_value = instances
    res = self.instance_group._resolve_attribute('InstanceList')
    self.assertEqual('2.1.3.1,2.1.3.2,2.1.3.3', res)