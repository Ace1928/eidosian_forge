import random
from unittest import mock
import uuid
from cinderclient import api_versions
from openstack.block_storage.v3 import _proxy
from openstack.block_storage.v3 import availability_zone as _availability_zone
from openstack.block_storage.v3 import extension as _extension
from openstack.block_storage.v3 import resource_filter as _filters
from openstack.block_storage.v3 import volume as _volume
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_v2_fakes
def create_one_volume_group(attrs=None):
    """Create a fake group.

    :param attrs: A dictionary with all attributes of group
    :return: A FakeResource object with id, name, status, etc.
    """
    attrs = attrs or {}
    group_type = attrs.pop('group_type', None) or uuid.uuid4().hex
    volume_types = attrs.pop('volume_types', None) or [uuid.uuid4().hex]
    group_info = {'id': uuid.uuid4().hex, 'status': random.choice(['available']), 'availability_zone': f'az-{uuid.uuid4().hex}', 'created_at': '2015-09-16T09:28:52.000000', 'name': 'first_group', 'description': f'description-{uuid.uuid4().hex}', 'group_type': group_type, 'volume_types': volume_types, 'volumes': [f'volume-{uuid.uuid4().hex}'], 'group_snapshot_id': None, 'source_group_id': None, 'project_id': f'project-{uuid.uuid4().hex}'}
    group_info.update(attrs)
    group = fakes.FakeResource(None, group_info, loaded=True)
    return group