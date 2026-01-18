import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_types(attrs=None, count=2):
    """Create multiple fake share types.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share types to be faked
        :return:
            A list of FakeResource objects
        """
    share_types = []
    for n in range(0, count):
        share_types.append(FakeShareType.create_one_sharetype(attrs))
    return share_types