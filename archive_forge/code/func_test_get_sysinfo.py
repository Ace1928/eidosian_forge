import os
import platform
import re
import shutil
import tempfile
import subprocess
import pyomo.common.unittest as unittest
import pyomo.common.envvar as envvar
from pyomo.common import DeveloperError
from pyomo.common.fileutils import this_file
from pyomo.common.download import FileDownloader, distro_available
from pyomo.common.tee import capture_output
def test_get_sysinfo(self):
    f = FileDownloader()
    ans = f.get_sysinfo()
    self.assertIs(type(ans), tuple)
    self.assertEqual(len(ans), 2)
    self.assertTrue(len(ans[0]) > 0)
    self.assertTrue(platform.system().lower().startswith(ans[0]))
    self.assertFalse(any((c in ans[0] for c in '.-_')))
    self.assertIn(ans[1], (32, 64))