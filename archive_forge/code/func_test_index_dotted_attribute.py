import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_index_dotted_attribute(self):
    mock_members = self.patchobject(grouputils, 'get_members')
    self.group.nested = mock.Mock()
    members = []
    output = []
    for ip_ex in range(0, 2):
        inst = mock.Mock()
        inst.name = 'ab'[ip_ex]
        inst.FnGetAtt.return_value = '2.1.3.%d' % ip_ex
        output.append('2.1.3.%d' % ip_ex)
        members.append(inst)
    mock_members.return_value = members
    self.assertEqual(output[0], self.group.FnGetAtt('resource.0', 'Bar'))
    self.assertEqual(output[1], self.group.FnGetAtt('resource.1.Bar'))
    self.assertRaises(exception.NotFound, self.group.FnGetAtt, 'resource.2')