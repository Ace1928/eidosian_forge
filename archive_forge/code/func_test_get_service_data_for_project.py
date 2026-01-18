import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_get_service_data_for_project(self):
    if self.is_secondary:
        self.skipTest('Secondary services have no project mapping')
        return
    elif not self.project:
        self.skipTest('Empty project is invalid but tested elsewhere.')
        return
    service_data = self.service_types.get_service_data_for_project(self.project)
    api_url = 'https://developer.openstack.org/api-ref/{api_reference}/'
    self.assertIsNotNone(service_data)
    if self.api_reference_project:
        self.assertEqual(self.api_reference_project, service_data['api_reference_project'])
    else:
        self.assertEqual(self.project, service_data['project'])
    self.assertEqual(self.official, service_data['service_type'])
    self.assertEqual(api_url.format(api_reference=self.api_reference), service_data['api_reference'])