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
def test_getlogin_bad(self):
    with mock.patch('os.getenv') as os_getenv:
        with mock.patch('os.getlogin') as os_getlogin:
            os_getenv.side_effect = [None, None, 'bar']
            os_getlogin.side_effect = OSError('[Errno 22] Invalid argument')
            self.assertEqual('bar', wrapper._getlogin())
            os_getlogin.assert_called_once_with()
            self.assertEqual(3, os_getenv.call_count)