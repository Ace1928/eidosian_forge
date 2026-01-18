import copy
import random
from unittest import mock
import uuid
from cinderclient import api_versions
from openstack.block_storage.v2 import _proxy as block_storage_v2_proxy
from openstack.block_storage.v3 import backup as _backup
from openstack.block_storage.v3 import capabilities as _capabilities
from openstack.block_storage.v3 import stats as _stats
from openstack.block_storage.v3 import volume as _volume
from openstack.image.v2 import _proxy as image_v2_proxy
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def create_one_volume(attrs=None):
    """Create a fake volume.

    :param dict attrs:
        A dictionary with all attributes of volume
    :return:
        A FakeResource object with id, name, status, etc.
    """
    attrs = attrs or {}
    volume_info = {'id': 'volume-id' + uuid.uuid4().hex, 'name': 'volume-name' + uuid.uuid4().hex, 'description': 'description' + uuid.uuid4().hex, 'status': random.choice(['available', 'in_use']), 'size': random.randint(1, 20), 'volume_type': random.choice(['fake_lvmdriver-1', 'fake_lvmdriver-2']), 'bootable': random.randint(0, 1), 'metadata': {'key' + uuid.uuid4().hex: 'val' + uuid.uuid4().hex, 'key' + uuid.uuid4().hex: 'val' + uuid.uuid4().hex, 'key' + uuid.uuid4().hex: 'val' + uuid.uuid4().hex}, 'snapshot_id': random.randint(1, 5), 'availability_zone': 'zone' + uuid.uuid4().hex, 'attachments': [{'device': '/dev/' + uuid.uuid4().hex, 'server_id': uuid.uuid4().hex}]}
    volume_info.update(attrs)
    volume = fakes.FakeResource(None, volume_info, loaded=True)
    return volume