import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_one_project(attrs=None):
    """Create a fake project.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, name, and so on
        """
    attrs = attrs or {}
    project_info = {'id': 'project-id-' + uuid.uuid4().hex, 'name': 'project-name-' + uuid.uuid4().hex, 'description': 'project_description', 'enabled': True}
    project_info.update(attrs)
    project = fakes.FakeResource(info=copy.deepcopy(project_info), loaded=True)
    return project