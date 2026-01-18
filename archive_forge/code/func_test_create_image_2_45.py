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
def test_create_image_2_45(self):
    """Tests the image-create command with microversion 2.45 which
        does not change the output of the command, just how the response
        from the server is processed.
        """
    self.run_command('image-create sample-server mysnapshot', api_version='2.45')
    self.assert_called('POST', '/servers/1234/action', {'createImage': {'name': 'mysnapshot', 'metadata': {}}})