import os
import tempfile
import unittest
from .config_exception import ConfigException
from .incluster_config import (SERVICE_HOST_ENV_NAME, SERVICE_PORT_ENV_NAME,
def test_no_cert_file(self):
    loader = self.get_test_loader(cert_filename='not_exists_file_1123')
    self._should_fail_load(loader, 'cert file does not exists')