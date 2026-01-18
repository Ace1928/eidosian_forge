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
def create_volume_messages(attrs=None, count=2):
    """Create multiple fake messages.

    :param attrs: A dictionary with all attributes of message
    :param count: The number of messages to be faked
    :return: A list of FakeResource objects
    """
    messages = []
    for n in range(0, count):
        messages.append(create_one_volume_message(attrs))
    return messages