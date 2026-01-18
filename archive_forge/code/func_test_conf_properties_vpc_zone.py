import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_conf_properties_vpc_zone(self):
    self.stub_ImageConstraint_validate()
    self.stub_FlavorConstraint_validate()
    self.stub_SnapshotConstraint_validate()
    t = template_format.parse(as_template)
    properties = t['Resources']['WebServerGroup']['Properties']
    properties['VPCZoneIdentifier'] = ['xxxx']
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    conf = stack['LaunchConfig']
    self.assertIsNone(conf.validate())
    scheduler.TaskRunner(conf.create)()
    self.assertEqual((conf.CREATE, conf.COMPLETE), conf.state)
    group = stack['WebServerGroup']
    config, props = group._get_conf_properties()
    self.assertEqual('xxxx', props['SubnetId'])
    conf.delete()