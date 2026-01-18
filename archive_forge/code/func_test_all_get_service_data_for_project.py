import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_all_get_service_data_for_project(self):
    if not self.project:
        self.skipTest('Empty project is invalid but tested elsewhere.')
        return
    all_data = self.service_types.get_all_service_data_for_project(self.project)
    for index, data in enumerate(all_data):
        self.assertEqual(data, self.service_types.get_service_data(self.all_services[index]))