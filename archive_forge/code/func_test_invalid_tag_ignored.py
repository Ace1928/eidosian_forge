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
def test_invalid_tag_ignored(self):
    self.repo.commit()
    self.repo.tag('1')
    self.repo.commit()
    self.repo.tag('badver')
    version = packaging._get_version_from_git()
    self.assertThat(version, matchers.StartsWith('1.0.1.dev1'))
    self.repo.commit()
    version = packaging._get_version_from_git()
    self.assertThat(version, matchers.StartsWith('1.0.1.dev2'))
    self.repo.commit()
    self.repo.tag('1.2')
    self.repo.commit()
    self.repo.tag('badver2')
    version = packaging._get_version_from_git()
    self.assertThat(version, matchers.StartsWith('1.2.1.dev1'))
    self.repo.commit()
    self.repo.tag('1.2.3')
    self.repo.commit()
    self.repo.tag('badver3')
    version = packaging._get_version_from_git()
    self.assertThat(version, matchers.StartsWith('1.2.4.dev1'))
    self.repo.commit()
    self.repo.tag('1.2.4.0a1')
    self.repo.commit()
    self.repo.tag('badver4')
    version = packaging._get_version_from_git()
    self.assertThat(version, matchers.StartsWith('1.2.4.0a2.dev1'))
    self.repo.commit()
    self.repo.tag('2')
    self.repo.commit()
    self.repo.tag('non-release-tag/2014.12.16-1')
    version = packaging._get_version_from_git()
    self.assertThat(version, matchers.StartsWith('2.0.1.dev1'))