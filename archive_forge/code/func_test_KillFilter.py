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
def test_KillFilter(self):
    if not os.path.exists('/proc/%d' % os.getpid()):
        self.skipTest('Test requires /proc filesystem (procfs)')
    p = subprocess.Popen(['cat'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        f = filters.KillFilter('root', '/bin/cat', '-9', '-HUP')
        f2 = filters.KillFilter('root', '/usr/bin/cat', '-9', '-HUP')
        f3 = filters.KillFilter('root', '/usr/bin/coreutils', '-9', '-HUP')
        usercmd = ['kill', '-ALRM', p.pid]
        self.assertFalse(f.match(usercmd) or f2.match(usercmd))
        usercmd = ['kill', p.pid]
        self.assertFalse(f.match(usercmd) or f2.match(usercmd))
        usercmd = ['kill', '-9', p.pid]
        self.assertTrue(f.match(usercmd) or f2.match(usercmd) or f3.match(usercmd))
        f = filters.KillFilter('root', '/bin/cat')
        f2 = filters.KillFilter('root', '/usr/bin/cat')
        f3 = filters.KillFilter('root', '/usr/bin/coreutils')
        usercmd = ['kill', os.getpid()]
        self.assertFalse(f.match(usercmd) or f2.match(usercmd))
        usercmd = ['kill', 999999]
        self.assertFalse(f.match(usercmd) or f2.match(usercmd))
        usercmd = ['kill', p.pid]
        self.assertTrue(f.match(usercmd) or f2.match(usercmd) or f3.match(usercmd))
        f = filters.KillFilter('root', 'cat')
        f2 = filters.KillFilter('root', 'coreutils')
        usercmd = ['kill', os.getpid()]
        self.assertFalse(f.match(usercmd))
        usercmd = ['kill', p.pid]
        self.assertTrue(f.match(usercmd) or f2.match(usercmd))
        with fixtures.EnvironmentVariable('PATH', '/foo:/bar'):
            self.assertFalse(f.match(usercmd))
        with fixtures.EnvironmentVariable('PATH'):
            self.assertFalse(f.match(usercmd))
    finally:
        p.terminate()
        p.wait()