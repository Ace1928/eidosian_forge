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
def convert_all_json_to_yaml(self, dirpath):
    for path in os.listdir(dirpath):
        if not path.endswith('.template') and (not path.endswith('.json')):
            continue
        with open(os.path.join(dirpath, path), 'r') as f:
            json_str = f.read()
        yml_str = template_format.convert_json_to_yaml(json_str)
        yield (json_str, yml_str)