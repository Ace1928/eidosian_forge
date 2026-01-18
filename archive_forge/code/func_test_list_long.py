import json
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_list_long(self):
    """Check baremetal deploy template list --long command

        Test steps:
        1) Create baremetal deploy template in setUp.
        2) List baremetal deploy templates with detail=True.
        3) Check deploy template fields in output.
        """
    template_list = self.deploy_template_list(params='--long')
    template = [template for template in template_list if template['Name'] == self.template['name']][0]
    self.assertEqual(self.template['extra'], template['Extra'])
    self.assertEqual(self.template['name'], template['Name'])
    self.assertEqual(self.template['steps'], template['Steps'])
    self.assertEqual(self.template['uuid'], template['UUID'])