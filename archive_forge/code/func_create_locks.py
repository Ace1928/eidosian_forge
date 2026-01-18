import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_locks(attrs=None, count=2):
    """Create multiple fake locks.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share transfers to be faked
        :return:
            A list of FakeResource objects
        """
    resource_locks = []
    for n in range(0, count):
        resource_locks.append(FakeResourceLock.create_one_lock(attrs))
    return resource_locks