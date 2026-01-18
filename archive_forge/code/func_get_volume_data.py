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
def get_volume_data(volume=None):
    """Get the volume data from a faked volume object.

    :param volume:
        A FakeResource objects faking volume
    :return
        A tuple which may include the following values:
        ('ce26708d', 'fake_volume', 'fake description', 'available',
         20, 'fake_lvmdriver-1', "Alpha='a', Beta='b', Gamma='g'",
         1, 'nova', [{'device': '/dev/ice', 'server_id': '1233'}])
    """
    data_list = []
    if volume is not None:
        for x in sorted(volume.keys()):
            if x == 'tags':
                data_list.append(format_columns.ListColumn(volume.info.get(x)))
            else:
                data_list.append(volume.info.get(x))
    return tuple(data_list)