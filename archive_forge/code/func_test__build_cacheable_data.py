import sys
from unittest import mock
from stevedore import _cache
from stevedore.tests import utils
def test__build_cacheable_data(self):
    ret = _cache._build_cacheable_data()
    self.assertIsInstance(ret['groups'], dict)