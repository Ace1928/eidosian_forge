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
def test_list_sortby_index_with_sort(self):
    for cmd in ['list --sort key', 'list --sort key:desc', 'list --sort key1,key2:asc']:
        with mock.patch('novaclient.utils.print_list') as mock_print_list:
            self.run_command(cmd)
            mock_print_list.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY, sortby_index=None)