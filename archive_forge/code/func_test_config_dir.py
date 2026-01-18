import os
import sys
from .. import bedding, osutils, tests
def test_config_dir(self):
    self.assertIsSameRealPath(bedding.config_dir(), self.appdata_bzr)