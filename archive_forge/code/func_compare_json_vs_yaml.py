import os
from unittest import mock
import re
import yaml
from heat.common import config
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.tests import common
from heat.tests import utils
def compare_json_vs_yaml(self, json_str, yml_str):
    yml = template_format.parse(yml_str)
    self.assertEqual(u'2012-12-12', yml[u'HeatTemplateFormatVersion'])
    self.assertNotIn(u'AWSTemplateFormatVersion', yml)
    del yml[u'HeatTemplateFormatVersion']
    jsn = template_format.parse(json_str)
    if u'AWSTemplateFormatVersion' in jsn:
        del jsn[u'AWSTemplateFormatVersion']
    self.assertEqual(yml, jsn)