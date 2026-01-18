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
def create_one_domain(attrs=None):
    """Create a fake domain.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, name, and so on
        """
    attrs = attrs or {}
    domain_info = {'id': 'domain-id-' + uuid.uuid4().hex, 'name': 'domain-name-' + uuid.uuid4().hex, 'description': 'domain-description-' + uuid.uuid4().hex, 'enabled': True, 'tags': [], 'links': 'links-' + uuid.uuid4().hex}
    domain_info.update(attrs)
    domain = fakes.FakeResource(info=copy.deepcopy(domain_info), loaded=True)
    return domain