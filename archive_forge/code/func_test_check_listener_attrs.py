from osc_lib import exceptions
from osc_lib.tests import utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import validate
def test_check_listener_attrs(self):
    attrs_dict = {'protocol_port': constants.MIN_PORT_NUMBER}
    validate.check_listener_attrs(attrs_dict)
    attrs_dict = {'protocol_port': constants.MAX_PORT_NUMBER}
    validate.check_listener_attrs(attrs_dict)
    attrs_dict = {'protocol_port': constants.MIN_PORT_NUMBER - 1}
    self.assertRaises(exceptions.InvalidValue, validate.check_listener_attrs, attrs_dict)
    attrs_dict = {'protocol_port': constants.MAX_PORT_NUMBER + 1}
    self.assertRaises(exceptions.InvalidValue, validate.check_listener_attrs, attrs_dict)
    for key in ('hsts_preload', 'hsts_include_subdomains'):
        attrs_dict = {key: True}
        self.assertRaises(exceptions.InvalidValue, validate.check_listener_attrs, attrs_dict)