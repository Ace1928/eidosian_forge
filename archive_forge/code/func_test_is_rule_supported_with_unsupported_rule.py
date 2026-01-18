from unittest import mock
from neutron_lib.api.definitions import portbindings
from neutron_lib import constants
from neutron_lib.services.qos import base as qos_base
from neutron_lib.services.qos import constants as qos_consts
from neutron_lib.tests import _base
def test_is_rule_supported_with_unsupported_rule(self):
    self.assertFalse(_make_driver().is_rule_supported(_make_rule()))