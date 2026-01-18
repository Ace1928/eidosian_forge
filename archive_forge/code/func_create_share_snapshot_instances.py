import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_snapshot_instances(attrs=None, count=2):
    """Create multiple fake snapshot instances.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share snapshot instance locations to be faked
        :return:
            A list of FakeResource objects
        """
    share_snapshot_instances = []
    for n in range(0, count):
        share_snapshot_instances.append(FakeShareSnapshot.create_one_snapshot(attrs))
    return share_snapshot_instances