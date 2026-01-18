import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_instances(attrs=None, count=2):
    """Create multiple fake instances.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share instances to be faked
        :return:
            A list of FakeResource objects
        """
    share_instances = []
    for n in range(count):
        share_instances.append(FakeShareInstance.create_one_share_instance(attrs))
    return share_instances