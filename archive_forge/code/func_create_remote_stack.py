import collections
import json
from unittest import mock
from heatclient import exc
from heatclient.v1 import stacks
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import remote_stack
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common as tests_common
from heat.tests import utils
def create_remote_stack(self, stack_template=None):
    defaults = [get_stack(stack_status='CREATE_IN_PROGRESS'), get_stack(stack_status='CREATE_COMPLETE')]
    if self.parent is None:
        self.initialize(stack_template=stack_template)
    self.heat.stacks.create.return_value = {'stack': get_stack().to_dict()}
    self.heat.stacks.get = mock.MagicMock(side_effect=defaults)
    rsrc = self.parent['remote_stack']
    scheduler.TaskRunner(rsrc.create)()
    return rsrc