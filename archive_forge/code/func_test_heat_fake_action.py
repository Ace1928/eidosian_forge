import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
def test_heat_fake_action(self):
    self.assertRaises(exceptions.CommandFailed, self.heat, 'this-does-not-exist')