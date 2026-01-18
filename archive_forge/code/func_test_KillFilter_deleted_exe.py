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
def test_KillFilter_deleted_exe(self):
    """Makes sure deleted exe's are killed correctly."""
    command = '/bin/commandddddd'
    f = filters.KillFilter('root', command)
    usercmd = ['kill', 1234]
    with mock.patch('os.readlink') as readlink:
        readlink.return_value = command + ' (deleted)'
        with mock.patch('os.path.isfile') as exists:

            def fake_exists(path):
                return path == command
            exists.side_effect = fake_exists
            self.assertTrue(f.match(usercmd))