import os
import re
import unittest
from distutils import debug
from distutils.log import WARN
from distutils.errors import DistutilsTemplateError
from distutils.filelist import glob_to_re, translate_pattern, FileList
from distutils import filelist
from test.support import os_helper
from test.support import captured_stdout
from distutils.tests import support
def test_non_local_discovery(self):
    """
        When findall is called with another path, the full
        path name should be returned.
        """
    with os_helper.temp_dir() as temp_dir:
        file1 = os.path.join(temp_dir, 'file1.txt')
        os_helper.create_empty_file(file1)
        expected = [file1]
        self.assertEqual(filelist.findall(temp_dir), expected)