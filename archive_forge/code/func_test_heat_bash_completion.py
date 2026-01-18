import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
def test_heat_bash_completion(self):
    self.heat('bash-completion')