import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_servers(attrs=None, count=2):
    """Create multiple fake servers.

        :param dict attrs:
            A dictionary with all attributes
        :param int count:
            The number of share server to be faked
        :return:
            A list of FakeResource objects
        """
    attrs = attrs or {}
    share_servers = []
    for n in range(count):
        share_servers.append(FakeShareServer.create_one_server(attrs))
    return share_servers