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
def get_volume_columns(volume=None):
    """Get the volume columns from a faked volume object.

    :param volume:
        A FakeResource objects faking volume
    :return
        A tuple which may include the following keys:
        ('id', 'name', 'description', 'status', 'size', 'volume_type',
         'metadata', 'snapshot', 'availability_zone', 'attachments')
    """
    if volume is not None:
        return tuple((k for k in sorted(volume.keys())))
    return tuple([])