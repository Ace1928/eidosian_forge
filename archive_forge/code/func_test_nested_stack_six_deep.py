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
def test_nested_stack_six_deep(self, tr):
    tmpl = "\nHeatTemplateFormatVersion: 2012-12-12\nResources:\n    Nested:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth%i.template'\n"
    root_template = tmpl % 1
    depth1_template = tmpl % 2
    depth2_template = tmpl % 3
    depth3_template = tmpl % 4
    depth4_template = tmpl % 5
    depth5_template = tmpl % 6
    depth5_template += '\n            Parameters:\n                KeyName: foo\n'
    urlfetch.get.side_effect = [depth1_template, depth2_template, depth3_template, depth4_template, depth5_template, self.nested_template]
    tr.return_value = 5
    t = template_format.parse(root_template)
    stack = self.parse_stack(t)
    stack['Nested'].root_stack_id = '1234'
    res = self.assertRaises(exception.StackValidationFailed, stack.validate)
    self.assertIn('Recursion depth exceeds', str(res))
    calls = [mock.call('https://server.test/depth1.template'), mock.call('https://server.test/depth2.template'), mock.call('https://server.test/depth3.template'), mock.call('https://server.test/depth4.template'), mock.call('https://server.test/depth5.template'), mock.call('https://server.test/depth6.template')]
    urlfetch.get.assert_has_calls(calls)