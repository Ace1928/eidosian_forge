from osc_lib import exceptions
from osc_lib.tests import utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import validate
def test_check_l7policy_attrs(self):
    attrs_dict = {'action': 'redirect_to_pool'.upper(), 'redirect_pool_id': 'id'}
    try:
        validate.check_l7policy_attrs(attrs_dict)
    except exceptions.CommandError as e:
        self.fail('%s raised unexpectedly' % e)
    attrs_dict.pop('redirect_pool_id')
    self.assertRaises(exceptions.CommandError, validate.check_l7policy_attrs, attrs_dict)
    attrs_dict = {'action': 'redirect_to_url'.upper(), 'redirect_url': 'url'}
    try:
        validate.check_l7policy_attrs(attrs_dict)
    except exceptions.CommandError as e:
        self.fail('%s raised unexpectedly' % e)
    attrs_dict.pop('redirect_url')
    self.assertRaises(exceptions.CommandError, validate.check_l7policy_attrs, attrs_dict)
    attrs_dict = {'action': 'redirect_prefix'.upper(), 'redirect_prefix': 'prefix'}
    try:
        validate.check_l7policy_attrs(attrs_dict)
    except exceptions.CommandError as e:
        self.fail('%s raised unexpectedly' % e)
    attrs_dict.pop('redirect_prefix')
    self.assertRaises(exceptions.CommandError, validate.check_l7policy_attrs, attrs_dict)