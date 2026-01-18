import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
def test_heat_resource_template_fmt_default(self):
    ret = self.heat('resource-template OS::Nova::Server')
    self.assertIn('Type: OS::Nova::Server', ret)