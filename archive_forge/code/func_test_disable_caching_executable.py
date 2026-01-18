import sys
from unittest import mock
from stevedore import _cache
from stevedore.tests import utils
def test_disable_caching_executable(self):
    """Test caching is disabled if python interpreter is located under /tmp
        directory (Ansible)
        """
    with mock.patch.object(sys, 'executable', '/tmp/fake'):
        sot = _cache.Cache()
        self.assertTrue(sot._disable_caching)