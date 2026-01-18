import os
import tempfile
import unittest
from .config_exception import ConfigException
from .incluster_config import (SERVICE_HOST_ENV_NAME, SERVICE_PORT_ENV_NAME,
def test_empty_host(self):
    loader = self.get_test_loader(environ={SERVICE_HOST_ENV_NAME: '', SERVICE_PORT_ENV_NAME: _TEST_PORT})
    self._should_fail_load(loader, 'empty host specified')