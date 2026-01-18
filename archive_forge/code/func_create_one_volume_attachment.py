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
def create_one_volume_attachment(attrs=None):
    """Create a fake volume attachment.

    :param attrs: A dictionary with all attributes of volume attachment
    :return: A FakeResource object with id, status, etc.
    """
    attrs = attrs or {}
    attachment_id = uuid.uuid4().hex
    volume_id = attrs.pop('volume_id', None) or uuid.uuid4().hex
    server_id = attrs.pop('instance', None) or uuid.uuid4().hex
    attachment_info = {'id': attachment_id, 'volume_id': volume_id, 'instance': server_id, 'status': random.choice(['attached', 'attaching', 'detached', 'reserved', 'error_attaching', 'error_detaching', 'deleted']), 'attach_mode': random.choice(['ro', 'rw']), 'attached_at': '2015-09-16T09:28:52.000000', 'detached_at': None, 'connection_info': {'access_mode': 'rw', 'attachment_id': attachment_id, 'auth_method': 'CHAP', 'auth_password': 'AcUZ8PpxLHwzypMC', 'auth_username': '7j3EZQWT3rbE6pcSGKvK', 'cacheable': False, 'driver_volume_type': 'iscsi', 'encrypted': False, 'qos_specs': None, 'target_discovered': False, 'target_iqn': f'iqn.2010-10.org.openstack:volume-{attachment_id}', 'target_lun': '1', 'target_portal': '192.168.122.170:3260', 'volume_id': volume_id}}
    attachment_info.update(attrs)
    return fakes.FakeResource(None, attachment_info, loaded=True)