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
def create_one_resource_filter(attrs=None):
    """Create a fake resource filter.

    :param attrs: A dictionary with all attributes of resource filter
    :return: A FakeResource object with id, name, status, etc.
    """
    attrs = attrs or {}
    resource_filter_info = {'filters': ['name', 'status', 'image_metadata', 'bootable', 'migration_status'], 'resource': 'volume'}
    resource_filter_info.update(attrs)
    return _filters.ResourceFilter(**resource_filter_info)