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
def create_launch_config(self, t, stack):
    self.stub_ImageConstraint_validate()
    self.stub_FlavorConstraint_validate()
    self.stub_SnapshotConstraint_validate()
    rsrc = stack['LaunchConfig']
    self.assertIsNone(rsrc.validate())
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    return rsrc