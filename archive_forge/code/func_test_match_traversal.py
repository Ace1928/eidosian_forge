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
def test_match_traversal(self):
    args = ['chown', 'nova', self.TRAVERSAL_WITHIN_DIR]
    self.assertTrue(self.f.match(args))