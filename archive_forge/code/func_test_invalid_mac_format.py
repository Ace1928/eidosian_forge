from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_invalid_mac_format(self):
    invalidate_format = ['8.8.8.8', '0a-1b-3c-4d-5e-6f-1f', '0a-1b-3c-4d-5e-xx']
    for mac in invalidate_format:
        self.assertFalse(self.constraint.validate(mac, None))