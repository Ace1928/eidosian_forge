import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_messages(attrs={}, count=2):
    """Create multiple fake messages.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share types to be faked
        :return:
            A list of FakeResource objects
        """
    messages = []
    for n in range(0, count):
        messages.append(FakeMessage.create_one_message(attrs))
    return messages