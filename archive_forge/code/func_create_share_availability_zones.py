import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_availability_zones(attrs=None, count=2):
    """Create multiple availability zones.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of availability zones to be faked
        :return:
            A list of FakeResource objects
        """
    availability_zones = []
    for n in range(0, count):
        availability_zones.append(FakeShareAvailabilityZones.create_one_availability_zone(attrs))
    return availability_zones