import io
import operator
import tempfile
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack import connection
from openstack import exceptions
from openstack.image.v1 import image as image_v1
from openstack.image.v2 import image
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(connection.Connection, 'search_images')
def test_get_image_not_found(self, mock_search):
    mock_search.return_value = []
    r = self.cloud.get_image('doesNotExist')
    self.assertIsNone(r)