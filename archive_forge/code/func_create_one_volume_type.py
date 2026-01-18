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
def create_one_volume_type(attrs=None, methods=None):
    """Create a fake volume type.

    :param dict attrs:
        A dictionary with all attributes
    :param dict methods:
        A dictionary with all methods
    :return:
        A FakeResource object with id, name, description, etc.
    """
    attrs = attrs or {}
    methods = methods or {}
    volume_type_info = {'id': 'type-id-' + uuid.uuid4().hex, 'name': 'type-name-' + uuid.uuid4().hex, 'description': 'type-description-' + uuid.uuid4().hex, 'extra_specs': {'foo': 'bar'}, 'is_public': True}
    volume_type_info.update(attrs)
    volume_type = fakes.FakeResource(info=copy.deepcopy(volume_type_info), methods=methods, loaded=True)
    return volume_type