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
def test_changelog(self):
    self.run_setup('sdist', allow_fail=False)
    filename = os.path.join(self.package_dir, 'ChangeLog')
    self.assertFalse(os.path.exists(filename))