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
def create_one_consistency_group(attrs=None):
    """Create a fake consistency group.

    :param dict attrs:
        A dictionary with all attributes
    :return:
        A FakeResource object with id, name, description, etc.
    """
    attrs = attrs or {}
    consistency_group_info = {'id': 'backup-id-' + uuid.uuid4().hex, 'name': 'backup-name-' + uuid.uuid4().hex, 'description': 'description-' + uuid.uuid4().hex, 'status': 'error', 'availability_zone': 'zone' + uuid.uuid4().hex, 'created_at': 'time-' + uuid.uuid4().hex, 'volume_types': ['volume-type1']}
    consistency_group_info.update(attrs)
    consistency_group = fakes.FakeResource(info=copy.deepcopy(consistency_group_info), loaded=True)
    return consistency_group