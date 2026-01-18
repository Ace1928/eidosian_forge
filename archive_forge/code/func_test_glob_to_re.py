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
def test_glob_to_re(self):
    sep = os.sep
    if os.sep == '\\':
        sep = re.escape(os.sep)
    for glob, regex in (('foo*', '(?s:foo[^%(sep)s]*)\\Z'), ('foo?', '(?s:foo[^%(sep)s])\\Z'), ('foo??', '(?s:foo[^%(sep)s][^%(sep)s])\\Z'), ('foo\\\\*', '(?s:foo\\\\\\\\[^%(sep)s]*)\\Z'), ('foo\\\\\\*', '(?s:foo\\\\\\\\\\\\[^%(sep)s]*)\\Z'), ('foo????', '(?s:foo[^%(sep)s][^%(sep)s][^%(sep)s][^%(sep)s])\\Z'), ('foo\\\\??', '(?s:foo\\\\\\\\[^%(sep)s][^%(sep)s])\\Z')):
        regex = regex % {'sep': sep}
        self.assertEqual(glob_to_re(glob), regex)