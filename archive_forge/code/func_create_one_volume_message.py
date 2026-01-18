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
def create_one_volume_message(attrs=None):
    """Create a fake message.

    :param attrs: A dictionary with all attributes of message
    :return: A FakeResource object with id, name, status, etc.
    """
    attrs = attrs or {}
    message_info = {'created_at': '2016-02-11T11:17:37.000000', 'event_id': f'VOLUME_{random.randint(1, 999999):06d}', 'guaranteed_until': '2016-02-11T11:17:37.000000', 'id': uuid.uuid4().hex, 'message_level': 'ERROR', 'request_id': f'req-{uuid.uuid4().hex}', 'resource_type': 'VOLUME', 'resource_uuid': uuid.uuid4().hex, 'user_message': f'message-{uuid.uuid4().hex}'}
    message_info.update(attrs)
    return fakes.FakeResource(None, message_info, loaded=True)