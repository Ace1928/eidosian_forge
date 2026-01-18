from osc_lib import exceptions
from osc_lib.tests import utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import validate
def test_check_member_attrs(self):
    attrs_dict = {'protocol_port': constants.MIN_PORT_NUMBER, 'member_port': constants.MIN_PORT_NUMBER, 'weight': constants.MIN_WEIGHT}
    validate.check_member_attrs(attrs_dict)
    attrs_dict = {'protocol_port': constants.MAX_PORT_NUMBER, 'member_port': constants.MAX_PORT_NUMBER, 'weight': constants.MAX_WEIGHT}
    validate.check_member_attrs(attrs_dict)
    attrs_dict = {'protocol_port': constants.MIN_PORT_NUMBER - 1, 'member_port': constants.MIN_PORT_NUMBER, 'weight': constants.MIN_WEIGHT}
    self.assertRaises(exceptions.InvalidValue, validate.check_member_attrs, attrs_dict)
    attrs_dict = {'protocol_port': constants.MAX_PORT_NUMBER + 1, 'member_port': constants.MIN_PORT_NUMBER, 'weight': constants.MIN_WEIGHT}
    self.assertRaises(exceptions.InvalidValue, validate.check_member_attrs, attrs_dict)
    attrs_dict = {'protocol_port': constants.MIN_PORT_NUMBER, 'member_port': constants.MIN_PORT_NUMBER - 1, 'weight': constants.MIN_WEIGHT}
    self.assertRaises(exceptions.InvalidValue, validate.check_member_attrs, attrs_dict)
    attrs_dict = {'protocol_port': constants.MIN_PORT_NUMBER, 'member_port': constants.MAX_PORT_NUMBER + 1, 'weight': constants.MIN_WEIGHT}
    self.assertRaises(exceptions.InvalidValue, validate.check_member_attrs, attrs_dict)
    attrs_dict = {'protocol_port': constants.MIN_PORT_NUMBER, 'member_port': constants.MIN_PORT_NUMBER, 'weight': constants.MIN_WEIGHT - 1}
    self.assertRaises(exceptions.InvalidValue, validate.check_member_attrs, attrs_dict)
    attrs_dict = {'protocol_port': constants.MIN_PORT_NUMBER, 'member_port': constants.MIN_PORT_NUMBER, 'weight': constants.MAX_WEIGHT + 1}
    self.assertRaises(exceptions.InvalidValue, validate.check_member_attrs, attrs_dict)