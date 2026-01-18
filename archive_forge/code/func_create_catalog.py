import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_catalog(attrs=None):
    """Create a fake catalog.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object with id, name, type and so on.
        """
    attrs = attrs or {}
    catalog_info = {'id': 'service-id-' + uuid.uuid4().hex, 'type': 'compute', 'name': 'supernova', 'endpoints': [{'region': 'one', 'publicURL': 'https://public.one.example.com', 'internalURL': 'https://internal.one.example.com', 'adminURL': 'https://admin.one.example.com'}, {'region': 'two', 'publicURL': 'https://public.two.example.com', 'internalURL': 'https://internal.two.example.com', 'adminURL': 'https://admin.two.example.com'}, {'region': None, 'publicURL': 'https://public.none.example.com', 'internalURL': 'https://internal.none.example.com', 'adminURL': 'https://admin.none.example.com'}]}
    catalog_info.update(attrs)
    catalog = fakes.FakeResource(info=copy.deepcopy(catalog_info), loaded=True)
    return catalog