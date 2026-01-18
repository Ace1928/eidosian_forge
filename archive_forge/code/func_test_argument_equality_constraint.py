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
def test_argument_equality_constraint(self):
    temp_file_path = os.path.join(self.tmp_root_dir, 'spam/eggs')
    f = filters.PathFilter('/bin/chown', 'root', 'nova', temp_file_path)
    args = ['chown', 'nova', temp_file_path]
    self.assertTrue(f.match(args))
    args = ['chown', 'quantum', temp_file_path]
    self.assertFalse(f.match(args))