import os
import tempfile
import unittest
from .config_exception import ConfigException
from .incluster_config import (SERVICE_HOST_ENV_NAME, SERVICE_PORT_ENV_NAME,
def test_empty_token_file(self):
    loader = self.get_test_loader(token_filename=self._create_file_with_temp_content())
    self._should_fail_load(loader, 'empty token file provided')