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
def create_one_volume_group_type(attrs=None, methods=None):
    """Create a fake group type.

    :param attrs: A dictionary with all attributes of group type
    :param methods: A dictionary with all methods
    :return: A FakeResource object with id, name, description, etc.
    """
    attrs = attrs or {}
    group_type_info = {'id': uuid.uuid4().hex, 'name': f'group-type-{uuid.uuid4().hex}', 'description': f'description-{uuid.uuid4().hex}', 'is_public': random.choice([True, False]), 'group_specs': {}}
    group_type_info.update(attrs)
    group_type = fakes.FakeResource(None, group_type_info, methods=methods, loaded=True)
    return group_type