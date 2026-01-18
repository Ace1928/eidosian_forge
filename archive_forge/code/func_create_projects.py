import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_projects(attrs=None, count=2):
    """Create multiple fake projects.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of projects to fake
        :return:
            A list of FakeResource objects faking the projects
        """
    projects = []
    for i in range(0, count):
        projects.append(FakeProject.create_one_project(attrs))
    return projects