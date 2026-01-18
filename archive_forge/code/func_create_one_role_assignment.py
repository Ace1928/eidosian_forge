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
def create_one_role_assignment(attrs=None):
    """Create a fake role assignment.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with scope, user, and so on
        """
    attrs = attrs or {}
    role_assignment_info = {'scope': {'project': {'id': 'project-id-' + uuid.uuid4().hex}}, 'user': {'id': 'user-id-' + uuid.uuid4().hex}, 'role': {'id': 'role-id-' + uuid.uuid4().hex}}
    role_assignment_info.update(attrs)
    role_assignment = fakes.FakeResource(info=copy.deepcopy(role_assignment_info), loaded=True)
    return role_assignment