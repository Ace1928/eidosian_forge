import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
def test_heat_template_function_list(self):
    ret = self.heat('template-function-list heat_template_version.2013-05-23')
    tmpl_functions = self.parser.listing(ret)
    self.assertTableStruct(tmpl_functions, ['functions', 'description'])