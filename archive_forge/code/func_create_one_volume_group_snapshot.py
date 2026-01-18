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
def create_one_volume_group_snapshot(attrs=None, methods=None):
    """Create a fake group snapshot.

    :param attrs: A dictionary with all attributes
    :param methods: A dictionary with all methods
    :return: A FakeResource object with id, name, description, etc.
    """
    attrs = attrs or {}
    group_snapshot_info = {'id': uuid.uuid4().hex, 'name': f'group-snapshot-{uuid.uuid4().hex}', 'description': f'description-{uuid.uuid4().hex}', 'status': random.choice(['available']), 'group_id': uuid.uuid4().hex, 'group_type_id': uuid.uuid4().hex, 'project_id': uuid.uuid4().hex}
    group_snapshot_info.update(attrs)
    group_snapshot = fakes.FakeResource(None, group_snapshot_info, methods=methods, loaded=True)
    return group_snapshot