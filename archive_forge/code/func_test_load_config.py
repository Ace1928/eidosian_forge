import os
import tempfile
import unittest
from .config_exception import ConfigException
from .incluster_config import (SERVICE_HOST_ENV_NAME, SERVICE_PORT_ENV_NAME,
def test_load_config(self):
    cert_filename = self._create_file_with_temp_content(_TEST_CERT)
    loader = self.get_test_loader(cert_filename=cert_filename)
    loader._load_config()
    self.assertEqual('https://' + _TEST_HOST_PORT, loader.host)
    self.assertEqual(cert_filename, loader.ssl_ca_cert)
    self.assertEqual(_TEST_TOKEN, loader.token)