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
def test_delete_autocreated_no_tasks(self):
    self.use_keystone_v3()
    self.cloud.image_api_use_tasks = False
    deleted = self.cloud.delete_autocreated_image_objects(container=self.container_name)
    self.assertFalse(deleted)
    self.assert_calls([])