import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def test_action_status(self):
    if not self.action_status:
        self.skipTest('API definition has no action status.')
    for status in self.action_status.values():
        self.assertIn(status, base.KNOWN_ACTION_STATUSES, 'HTTP status is unknown, check for typos.')