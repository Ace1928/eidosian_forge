import re
from unittest import mock
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_hidden_sequence_property(self):
    hidden_prop_templ = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  secret:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 100\n      sequence: octdigits\n        "
    stack = self.create_stack(hidden_prop_templ)
    secret = stack['secret']
    random_string = secret.FnGetAtt('value')
    self.assert_min('[0-7]', random_string, 100)
    self.assertEqual(secret.FnGetRefId(), random_string)
    self.assertIsNone(secret.properties['sequence'])
    expected = [{'class': u'octdigits', 'min': 1}]
    self.assertEqual(expected, secret.properties['character_classes'])