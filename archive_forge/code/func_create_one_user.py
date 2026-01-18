import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_one_user(attrs=None):
    """Create a fake user.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, name, and so on
        """
    attrs = attrs or {}
    user_info = {'id': 'user-id-' + uuid.uuid4().hex, 'name': 'user-name-' + uuid.uuid4().hex, 'tenantId': 'project-id-' + uuid.uuid4().hex, 'email': 'admin@openstack.org', 'enabled': True}
    user_info.update(attrs)
    user = fakes.FakeResource(info=copy.deepcopy(user_info), loaded=True)
    return user