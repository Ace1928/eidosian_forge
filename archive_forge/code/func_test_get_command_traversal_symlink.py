import configparser
import logging
import logging.handlers
import os
import tempfile
from unittest import mock
import uuid
import fixtures
import testtools
from oslo_rootwrap import cmd
from oslo_rootwrap import daemon
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def test_get_command_traversal_symlink(self):
    args = ['chown', 'nova', self.TRAVERSAL_SYMLINK_WITHIN_DIR]
    expected = ['/bin/chown', 'nova', os.path.realpath(self.TRAVERSAL_SYMLINK_WITHIN_DIR)]
    self.assertEqual(expected, self.f.get_command(args))