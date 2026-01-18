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
def test_handle_update_desired_cap(self):
    self.group._try_rolling_update = mock.Mock(return_value=None)
    self.group.resize = mock.Mock(return_value=None)
    props = {'DesiredCapacity': 4, 'MinSize': 0, 'MaxSize': 6}
    self.group._get_new_capacity = mock.Mock(return_value=4)
    defn = rsrc_defn.ResourceDefinition('nopayload', 'AWS::AutoScaling::AutoScalingGroup', props)
    self.group.handle_update(defn, None, props)
    self.group.resize.assert_called_once_with(4)
    self.group._try_rolling_update.assert_called_once_with(props)