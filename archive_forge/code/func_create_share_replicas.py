import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_replicas(attrs=None, count=2):
    """Create multiple fake replicas.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share types to be faked
        :return:
            A list of FakeResource objects
        """
    share_replicas = []
    for n in range(0, count):
        share_replicas.append(FakeShareReplica.create_one_replica(attrs))
    return share_replicas