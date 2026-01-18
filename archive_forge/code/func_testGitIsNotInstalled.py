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
def testGitIsNotInstalled(self):
    with mock.patch.object(git, '_run_shell_command') as _command:
        _command.side_effect = OSError
        self.assertEqual(False, git._git_is_installed())