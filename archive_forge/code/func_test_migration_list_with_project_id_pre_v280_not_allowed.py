import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
def test_migration_list_with_project_id_pre_v280_not_allowed(self):
    project_id = 'b59c18e5fa284fd384987c5cb25a1853'
    cmd = 'migration-list --project-id %s' % project_id
    self.assertRaises(SystemExit, self.run_command, cmd, api_version='2.79')