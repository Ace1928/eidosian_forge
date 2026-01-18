import contextlib
import copy
import json as jsonutils
import os
from unittest import mock
from cliff import columns as cliff_columns
import fixtures
from keystoneauth1 import loading
from openstack.config import cloud_region
from openstack.config import defaults
from oslo_utils import importutils
from requests_mock.contrib import fixture
import testtools
from osc_lib import clientmanager
from osc_lib import shell
from osc_lib.tests import fakes
def make_shell(shell_class=None):
    """Create a new command shell and mock out some bits."""
    if shell_class is None:
        shell_class = shell.OpenStackShell
    _shell = shell_class()
    _shell.command_manager = mock.Mock()
    return _shell