import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_network_subnets(attrs=None, count=2):
    """Create multiple fake share network subnets.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share network subnets to be faked
        :return:
            A list of FakeResource objects
        """
    share_network_subnets = []
    for n in range(count):
        share_network_subnets.append(FakeShareNetworkSubnet.create_one_share_subnet(attrs))
    return share_network_subnets