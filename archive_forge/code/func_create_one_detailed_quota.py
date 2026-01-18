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
def create_one_detailed_quota(attrs=None):
    """Create one quota"""
    attrs = attrs or {}
    quota_attrs = {'volumes': {'limit': 3, 'in_use': 1, 'reserved': 0}, 'per_volume_gigabytes': {'limit': -1, 'in_use': 0, 'reserved': 0}, 'snapshots': {'limit': 10, 'in_use': 0, 'reserved': 0}, 'gigabytes': {'limit': 1000, 'in_use': 5, 'reserved': 0}, 'backups': {'limit': 10, 'in_use': 0, 'reserved': 0}, 'backup_gigabytes': {'limit': 1000, 'in_use': 0, 'reserved': 0}, 'volumes_lvmdriver-1': {'limit': -1, 'in_use': 1, 'reserved': 0}, 'gigabytes_lvmdriver-1': {'limit': -1, 'in_use': 5, 'reserved': 0}, 'snapshots_lvmdriver-1': {'limit': -1, 'in_use': 0, 'reserved': 0}, 'volumes___DEFAULT__': {'limit': -1, 'in_use': 0, 'reserved': 0}, 'gigabytes___DEFAULT__': {'limit': -1, 'in_use': 0, 'reserved': 0}, 'snapshots___DEFAULT__': {'limit': -1, 'in_use': 0, 'reserved': 0}, 'groups': {'limit': 10, 'in_use': 0, 'reserved': 0}, 'id': uuid.uuid4().hex}
    quota_attrs.update(attrs)
    quota = fakes.FakeResource(info=copy.deepcopy(quota_attrs), loaded=True)
    return quota