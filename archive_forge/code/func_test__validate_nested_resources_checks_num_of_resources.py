import contextlib
import json
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import exceptions as msg_exceptions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources import stack_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test__validate_nested_resources_checks_num_of_resources(self):
    stack_resource.cfg.CONF.set_override('max_resources_per_stack', 2)
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'r': {'Type': 'OS::Heat::None'}}}
    template = stack_resource.template.Template(tmpl)
    root_resources = mock.Mock(return_value=2)
    self.parent_resource.stack.total_resources = root_resources
    self.assertRaises(exception.RequestLimitExceeded, self.parent_resource._validate_nested_resources, template)