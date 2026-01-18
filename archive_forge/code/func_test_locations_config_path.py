import os
import sys
from .. import bedding, osutils, tests
def test_locations_config_path(self):
    self.assertIsSameRealPath(bedding.locations_config_path(), self.appdata_bzr + '/locations.conf')
    self.overrideAttr(win32utils, 'get_appdata_location', lambda: None)
    self.assertRaises(RuntimeError, bedding.locations_config_path)