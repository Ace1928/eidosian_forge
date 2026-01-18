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
def test_IpNetnsExecFilter_match(self):
    f = filters.IpNetnsExecFilter(self._ip, 'root')
    self.assertTrue(f.match(['ip', 'netns', 'exec', 'foo', 'ip', 'link', 'list']))
    self.assertTrue(f.match(['ip', 'net', 'exec', 'foo', 'bar']))
    self.assertTrue(f.match(['ip', 'netn', 'e', 'foo', 'bar']))
    self.assertTrue(f.match(['ip', 'net', 'e', 'foo', 'bar']))
    self.assertTrue(f.match(['ip', 'net', 'exe', 'foo', 'bar']))