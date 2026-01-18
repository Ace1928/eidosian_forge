import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_pools(attrs=None, count=2):
    """Create multiple fake share pools.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share pools to be faked
        :return:
            A list of FakeResource objects
        """
    share_pools = []
    for n in range(count):
        share_pools.append(FakeSharePools.create_one_share_pool(attrs))
    return share_pools