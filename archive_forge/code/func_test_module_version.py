import testtools
import uuid
import barbicanclient
from barbicanclient import base
from barbicanclient import version
def test_module_version(self):
    self.assertTrue(hasattr(barbicanclient, '__version__'))
    self.assertTrue(hasattr(version, '__version__'))