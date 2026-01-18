import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_users(attrs=None, count=2):
    """Create multiple fake users.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of users to fake
        :return:
            A list of FakeResource objects faking the users
        """
    users = []
    for i in range(0, count):
        users.append(FakeUser.create_one_user(attrs))
    return users