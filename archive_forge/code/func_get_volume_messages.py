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
def get_volume_messages(messages=None, count=2):
    """Get an iterable MagicMock object with a list of faked messages.

    If messages list is provided, then initialize the Mock object with the
    list. Otherwise create one.

    :param messages: A list of FakeResource objects faking messages
    :param count: The number of messages to be faked
    :return An iterable Mock object with side_effect set to a list of faked
        messages
    """
    if messages is None:
        messages = create_volume_messages(count)
    return mock.Mock(side_effect=messages)