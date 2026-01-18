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
def create_cleanup_records():
    """Create fake service cleanup records.

    :return: A list of FakeResource objects
    """
    cleaning_records = []
    unavailable_records = []
    cleaning_work_info = {'id': 1, 'host': 'devstack@fakedriver-1', 'binary': 'cinder-volume', 'cluster_name': 'fake_cluster'}
    unavailable_work_info = {'id': 2, 'host': 'devstack@fakedriver-2', 'binary': 'cinder-scheduler', 'cluster_name': 'new_cluster'}
    cleaning_records.append(cleaning_work_info)
    unavailable_records.append(unavailable_work_info)
    cleaning = [fakes.FakeResource(None, obj, loaded=True) for obj in cleaning_records]
    unavailable = [fakes.FakeResource(None, obj, loaded=True) for obj in unavailable_records]
    return (cleaning, unavailable)