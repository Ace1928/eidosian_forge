import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_is_official(self):
    self.assertEqual(self.is_official, self.service_types.is_official(self.service_type))