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
def test_exec_dirs_search(self):
    f = filters.CommandFilter('cat', 'root')
    usercmd = ['cat', '/f']
    self.assertTrue(f.match(usercmd))
    self.assertTrue(f.get_command(usercmd, exec_dirs=['/bin', '/usr/bin']) in (['/bin/cat', '/f'], ['/usr/bin/cat', '/f']))