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
def test_instance_group_refid_rsrc_name(self):
    self.instance_group.id = '123'
    self.instance_group.uuid = '9bfb9456-3fe8-41f4-b318-9dba18eeef74'
    self.instance_group.action = 'CREATE'
    expected = self.instance_group.name
    self.assertEqual(expected, self.instance_group.FnGetRefId())