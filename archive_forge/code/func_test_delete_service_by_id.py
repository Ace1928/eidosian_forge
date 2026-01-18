import random
import string
from openstack.cloud import exc
from openstack import exceptions
from openstack.tests.functional import base
def test_delete_service_by_id(self):
    service = self.operator_cloud.create_service(name=self.new_service_name + '_delete_by_id', type='test_type')
    self.operator_cloud.delete_service(name_or_id=service['id'])
    observed_services = self.operator_cloud.list_services()
    found = False
    for s in observed_services:
        if s['id'] == service['id']:
            found = True
    self.assertEqual(False, found, message='service was not deleted!')