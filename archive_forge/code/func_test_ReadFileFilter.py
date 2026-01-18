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
def test_ReadFileFilter(self):
    goodfn = '/good/file.name'
    f = filters.ReadFileFilter(goodfn)
    usercmd = ['cat', '/bad/file']
    self.assertFalse(f.match(['cat', '/bad/file']))
    usercmd = ['cat', goodfn]
    self.assertEqual(['/bin/cat', goodfn], f.get_command(usercmd))
    self.assertTrue(f.match(usercmd))