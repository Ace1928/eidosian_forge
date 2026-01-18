import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
def test_index_present(self):
    tempdir = tempfile.mkdtemp()
    requirements = os.path.join(tempdir, 'requirements.txt')
    with open(requirements, 'w') as f:
        f.write('-i https://myindex.local\n')
        f.write('  --index-url https://myindex.local\n')
        f.write(' --extra-index-url https://myindex.local\n')
        f.write('--find-links https://myindex.local\n')
        f.write('arequirement>=1.0\n')
    result = packaging.parse_requirements([requirements])
    self.assertEqual(['arequirement>=1.0'], result)