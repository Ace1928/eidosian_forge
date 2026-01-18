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
def test_integer_only_keys_get_translated_correctly(self):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates/WordPress_Single_Instance.template')
    with open(path, 'r') as f:
        json_str = f.read()
        yml_str = template_format.convert_json_to_yaml(json_str)
        match = re.search('[\\s,{]\\d+\\s*:', yml_str)
        self.assertIsNone(match)