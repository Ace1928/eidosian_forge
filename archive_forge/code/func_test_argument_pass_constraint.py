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
def test_argument_pass_constraint(self):
    f = filters.PathFilter('/bin/chown', 'root', 'pass', 'pass')
    args = ['chown', 'something', self.SIMPLE_FILE_OUTSIDE_DIR]
    self.assertTrue(f.match(args))