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
def test_boot_from_volume_with_volume_type(self):
    """Tests creating a volume-backed server from a source image and
        specifying the type of volume to create with microversion 2.67.
        """
    self.run_command('boot --flavor 1 --block-device id=%s,source=image,dest=volume,size=1,bootindex=0,shutdown=remove,tag=foo,volume_type=lvm bfv-server' % FAKE_UUID_1, api_version='2.67')
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'bfv-server', 'block_device_mapping_v2': [{'uuid': FAKE_UUID_1, 'source_type': 'image', 'destination_type': 'volume', 'volume_size': '1', 'delete_on_termination': True, 'tag': 'foo', 'boot_index': '0', 'volume_type': 'lvm'}], 'networks': 'auto', 'imageRef': '', 'min_count': 1, 'max_count': 1}})