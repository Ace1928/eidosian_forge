import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_dict_str_conversion(self):
    dic = {'key1': 'value1', 'key2': 'value2'}
    self.assertEqual(dic, helpers.str2dict(helpers.dict2str(dic)))