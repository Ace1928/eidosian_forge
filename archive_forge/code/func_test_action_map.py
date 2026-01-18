import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def test_action_map(self):
    self.assertIsInstance(self.action_map, dict)
    if not self.action_map:
        self.skipTest('API definition has no action map.')
    for key in self.action_map:
        for action in self.action_map[key].values():
            self.assertIn(action, base.KNOWN_HTTP_ACTIONS, 'HTTP verb is unknown, check for typos.')