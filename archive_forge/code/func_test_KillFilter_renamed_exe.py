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
@mock.patch('os.readlink')
@mock.patch('os.path.isfile')
@mock.patch('os.path.exists')
@mock.patch('os.access')
def test_KillFilter_renamed_exe(self, mock_access, mock_exists, mock_isfile, mock_readlink):
    """Makes sure renamed exe's are killed correctly."""
    command = '/bin/commandddddd'
    f = filters.KillFilter('root', command)
    usercmd = ['kill', 1234]

    def fake_os_func(path, *args):
        return path == command
    mock_readlink.return_value = command + ';90bfb2 (deleted)'
    m = mock.mock_open(read_data=command)
    with mock.patch('builtins.open', m, create=True):
        mock_isfile.side_effect = fake_os_func
        mock_exists.side_effect = fake_os_func
        mock_access.side_effect = fake_os_func
        self.assertTrue(f.match(usercmd))