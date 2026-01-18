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
def test_child_template(self):
    self.instance_group._create_template = mock.Mock(return_value='tpl')
    self.assertEqual('tpl', self.instance_group.child_template())
    self.instance_group._create_template.assert_called_once_with(2)