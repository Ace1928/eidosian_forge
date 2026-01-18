import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_fake_security_services(attrs=None, count=2):
    """Create multiple fake security services.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share security services to be faked
        :return:
            A list of FakeResource objects
        """
    security_services = []
    for n in range(count):
        security_services.append(FakeShareSecurityService.create_fake_security_service(attrs))
    return security_services