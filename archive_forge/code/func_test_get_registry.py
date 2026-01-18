import logging
import subprocess
from os.path import join as pjoin
from unittest.mock import patch
from jupyterlab import commands
from .test_jupyterlab import AppHandlerTest
def test_get_registry(self):
    with patch('subprocess.check_output') as check_output:
        yarn_registry = 'https://private.yarn/manager'
        check_output.return_value = b'\n'.join([b'{"type":"info","data":"yarn config"}', b'{"type":"inspect","data":{"registry":"' + bytes(yarn_registry, 'utf-8') + b'"}}', b'{"type":"info","data":"npm config"}', b'{"type":"inspect","data":{"registry":"' + bytes(yarn_registry, 'utf-8') + b'"}}'])
        handler = commands.AppOptions()
        self.assertEqual(handler.registry, yarn_registry)