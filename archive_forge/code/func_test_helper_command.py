import logging
import os
import pipes
import platform
import sys
import tempfile
import time
from unittest import mock
import testtools
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep import priv_context
from oslo_privsep.tests import testctx
def test_helper_command(self):
    self.privsep_conf.privsep.helper_command = 'foo --bar'
    _, temp_path = tempfile.mkstemp()
    cmd = testctx.context.helper_command(temp_path)
    expected = ['foo', '--bar', '--privsep_context', testctx.context.pypath, '--privsep_sock_path', temp_path]
    self.assertEqual(expected, cmd)