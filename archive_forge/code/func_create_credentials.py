import copy
import datetime
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_credentials(attrs=None, count=2):
    """Create multiple fake credentials.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of credentials to fake
        :return:
            A list of FakeResource objects faking the credentials
        """
    credentials = []
    for i in range(0, count):
        credential = FakeCredential.create_one_credential(attrs)
        credentials.append(credential)
    return credentials