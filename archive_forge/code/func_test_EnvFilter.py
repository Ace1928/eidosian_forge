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
def test_EnvFilter(self):
    envset = ['A=/some/thing', 'B=somethingelse']
    envcmd = ['env'] + envset
    realcmd = ['sleep', '10']
    usercmd = envcmd + realcmd
    f = filters.EnvFilter('env', 'root', 'A=', 'B=ignored', 'sleep')
    self.assertTrue(f.match(envcmd + ['sleep']))
    self.assertTrue(f.match(envset + ['sleep']))
    self.assertFalse(f.match(envcmd + ['sleep2']))
    self.assertFalse(f.match(envset + ['sleep2']))
    self.assertTrue(f.match(usercmd))
    self.assertFalse(f.match([envcmd, 'C=ELSE']))
    self.assertFalse(f.match(['env', 'C=xx']))
    self.assertFalse(f.match(['env', 'A=xx']))
    self.assertFalse(f.match(realcmd))
    self.assertFalse(f.match(envcmd))
    self.assertFalse(f.match(envcmd[1:]))
    self.assertEqual(realcmd, f.exec_args(usercmd))
    env = f.get_environment(usercmd)
    self.assertEqual('/some/thing', env.get('A'))
    self.assertEqual('somethingelse', env.get('B'))
    self.assertNotIn('sleep', env.keys())