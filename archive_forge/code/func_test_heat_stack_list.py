import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
def test_heat_stack_list(self):
    self.heat('stack-list')