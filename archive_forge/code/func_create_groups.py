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
def create_groups(attrs=None, count=2):
    """Create multiple fake groups.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of groups to fake
        :return:
            A list of FakeResource objects faking the groups
        """
    groups = []
    for i in range(0, count):
        group = FakeGroup.create_one_group(attrs)
        groups.append(group)
    return groups