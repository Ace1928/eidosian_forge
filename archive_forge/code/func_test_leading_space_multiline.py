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
def test_leading_space_multiline(self):
    self.repo.commit()
    self.repo.tag('1.2.3')
    self.repo.commit('   Some cool text\n   sem-ver: api-break')
    version = packaging._get_version_from_git()
    self.assertThat(version, matchers.StartsWith('2.0.0.dev1'))