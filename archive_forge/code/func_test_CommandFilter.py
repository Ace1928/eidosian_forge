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
def test_CommandFilter(self):
    f = filters.CommandFilter('sleep', 'root', '10')
    self.assertFalse(f.match(['sleep2']))
    self.assertTrue(f.match(['sleep']))
    self.assertTrue(f.match(['sleep', 'anything']))
    self.assertTrue(f.match(['sleep', '10']))
    f = filters.CommandFilter('sleep', 'root')
    self.assertTrue(f.match(['sleep', '10']))