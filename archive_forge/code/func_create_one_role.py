import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_one_role(attrs=None):
    """Create a fake role.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, name, and so on
        """
    attrs = attrs or {}
    role_info = {'id': 'role-id' + uuid.uuid4().hex, 'name': 'role-name' + uuid.uuid4().hex}
    role_info.update(attrs)
    role = fakes.FakeResource(info=copy.deepcopy(role_info), loaded=True)
    return role