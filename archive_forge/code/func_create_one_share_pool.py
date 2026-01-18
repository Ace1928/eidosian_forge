import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_share_pool(attrs=None):
    """Create a fake share pool

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object
        """
    attrs = attrs or {}
    share_pool = {'name': 'fake_pool@gamma#fake_pool', 'host': 'fake_host_' + uuid.uuid4().hex, 'backend': 'fake_backend_' + uuid.uuid4().hex, 'pool': 'fake_pool_' + uuid.uuid4().hex, 'capabilities': {'fake_capability': uuid.uuid4().hex}}
    share_pool.update(attrs)
    share_pool = osc_fakes.FakeResource(info=copy.deepcopy(share_pool), loaded=True)
    return share_pool