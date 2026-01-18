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
def get_consistency_group_snapshots(snapshots=None, count=2):
    """Get an iterable MagicMock object with a list of faked cgsnapshots.

    If consistenct group snapshots list is provided, then initialize
    the Mock object with the list. Otherwise create one.

    :param List snapshots:
        A list of FakeResource objects faking consistency group snapshots
    :param Integer count:
        The number of consistency group snapshots to be faked
    :return
        An iterable Mock object with side_effect set to a list of faked
        consistency groups
    """
    if snapshots is None:
        snapshots = create_consistency_group_snapshots(count)
    return mock.Mock(side_effect=snapshots)