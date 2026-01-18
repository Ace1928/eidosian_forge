from osc_lib import exceptions
from osc_lib.tests import utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import validate
def test_check_l7rule_attrs(self):
    for i in ('cookie', 'header'):
        attrs_dict = {'type': i.upper(), 'key': 'key'}
        try:
            validate.check_l7rule_attrs(attrs_dict)
        except exceptions.CommandError as e:
            self.fail('%s raised unexpectedly' % e)
        attrs_dict.pop('key')
        self.assertRaises(exceptions.CommandError, validate.check_l7rule_attrs, attrs_dict)