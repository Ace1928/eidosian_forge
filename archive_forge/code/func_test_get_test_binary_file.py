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
def test_get_test_binary_file(self):
    tmpdir = tempfile.mkdtemp()
    try:
        f = FileDownloader()
        f.retrieve_url = lambda url: bytes('\n', encoding='utf-8')
        target = os.path.join(tmpdir, 'bin.txt')
        f.set_destination_filename(target)
        f.get_binary_file(None)
        self.assertEqual(os.path.getsize(target), 1)
        target = os.path.join(tmpdir, 'txt.txt')
        f.set_destination_filename(target)
        f.get_text_file(None)
        self.assertEqual(os.path.getsize(target), len(os.linesep))
    finally:
        shutil.rmtree(tmpdir)