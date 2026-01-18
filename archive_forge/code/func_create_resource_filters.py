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
def create_resource_filters(attrs=None, count=2):
    """Create multiple fake resource filters.

    :param attrs: A dictionary with all attributes of resource filter
    :param count: The number of resource filters to be faked
    :return: A list of FakeResource objects
    """
    resource_filters = []
    for n in range(0, count):
        resource_filters.append(create_one_resource_filter(attrs))
    return resource_filters