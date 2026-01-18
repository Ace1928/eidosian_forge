import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_get_service_type(self):
    if self.official:
        self.assertEqual(self.official, self.service_types.get_service_type(self.service_type))
    else:
        self.assertIsNone(self.service_types.get_service_type(self.service_type))