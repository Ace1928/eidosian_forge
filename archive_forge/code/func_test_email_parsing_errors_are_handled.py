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
def test_email_parsing_errors_are_handled(self):
    mocked_open = mock.mock_open()
    with mock.patch('pbr.packaging.open', mocked_open):
        with mock.patch('email.message_from_file') as message_from_file:
            message_from_file.side_effect = [email.errors.MessageError('Test'), {'Name': 'pbr_testpackage'}]
            version = packaging._get_version_from_pkg_metadata('pbr_testpackage')
    self.assertTrue(message_from_file.called)
    self.assertIsNone(version)