import os
import tempfile
import unittest
from .config_exception import ConfigException
from .incluster_config import (SERVICE_HOST_ENV_NAME, SERVICE_PORT_ENV_NAME,
def get_test_loader(self, token_filename=None, cert_filename=None, environ=_TEST_ENVIRON):
    if not token_filename:
        token_filename = self._create_file_with_temp_content(_TEST_TOKEN)
    if not cert_filename:
        cert_filename = self._create_file_with_temp_content(_TEST_CERT)
    return InClusterConfigLoader(token_filename=token_filename, cert_filename=cert_filename, environ=environ)