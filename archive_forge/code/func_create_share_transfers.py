import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_transfers(attrs=None, count=2):
    """Create multiple fake transfers.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share transfers to be faked
        :return:
            A list of FakeResource objects
        """
    share_transfers = []
    for n in range(0, count):
        share_transfers.append(FakeShareSnapshot.create_one_snapshot(attrs))
    return share_transfers