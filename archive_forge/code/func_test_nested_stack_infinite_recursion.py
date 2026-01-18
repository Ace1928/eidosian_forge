from unittest import mock
from oslo_config import cfg
from requests import exceptions
import yaml
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import api
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.cfn import stack as stack_res
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@mock.patch.object(parser.Stack, 'total_resources')
def test_nested_stack_infinite_recursion(self, tr):
    tmpl = "\nHeatTemplateFormatVersion: 2012-12-12\nResources:\n    Nested:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/the.template'\n"
    urlfetch.get.return_value = tmpl
    t = template_format.parse(tmpl)
    stack = self.parse_stack(t)
    stack['Nested'].root_stack_id = '1234'
    tr.return_value = 2
    res = self.assertRaises(exception.StackValidationFailed, stack.validate)
    self.assertIn('Recursion depth exceeds', str(res))
    expected_count = cfg.CONF.get('max_nested_stack_depth') + 1
    self.assertEqual(expected_count, urlfetch.get.call_count)