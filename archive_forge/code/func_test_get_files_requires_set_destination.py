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
def test_get_files_requires_set_destination(self):
    f = FileDownloader()
    with self.assertRaisesRegex(DeveloperError, 'target file name has not been initialized'):
        f.get_binary_file('bogus')
    with self.assertRaisesRegex(DeveloperError, 'target file name has not been initialized'):
        f.get_binary_file_from_zip_archive('bogus', 'bogus')
    with self.assertRaisesRegex(DeveloperError, 'target file name has not been initialized'):
        f.get_gzipped_binary_file('bogus')