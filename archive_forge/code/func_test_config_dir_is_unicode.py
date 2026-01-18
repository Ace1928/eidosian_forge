import os
import sys
from .. import bedding, osutils, tests
def test_config_dir_is_unicode(self):
    self.assertIsInstance(bedding.config_dir(), str)