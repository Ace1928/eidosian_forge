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
def test_boot_image_bdms_v2_no_destination_type(self):
    self.run_command('boot --flavor 1 --image %s --block-device id=fake-id,source=volume,device=vda,size=1,format=ext4,type=disk,shutdown=preserve some-server' % FAKE_UUID_1)
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'block_device_mapping_v2': [{'uuid': FAKE_UUID_1, 'source_type': 'image', 'destination_type': 'local', 'boot_index': 0, 'delete_on_termination': True}, {'uuid': 'fake-id', 'source_type': 'volume', 'destination_type': 'volume', 'device_name': 'vda', 'volume_size': '1', 'guest_format': 'ext4', 'device_type': 'disk', 'delete_on_termination': False}], 'imageRef': FAKE_UUID_1, 'min_count': 1, 'max_count': 1}})