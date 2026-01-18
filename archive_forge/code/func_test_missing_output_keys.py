import os
import unittest
import mock
from .config_exception import ConfigException
from .exec_provider import ExecProvider
from .kube_config import ConfigNode
@mock.patch('subprocess.Popen')
def test_missing_output_keys(self, mock):
    instance = mock.return_value
    instance.wait.return_value = 0
    outputs = ['\n            {\n                "kind": "ExecCredential",\n                "status": {\n                    "token": "dummy"\n                }\n            }\n            ', '\n            {\n                "apiVersion": "client.authentication.k8s.io/v1beta1",\n                "status": {\n                    "token": "dummy"\n                }\n            }\n            ', '\n            {\n                "apiVersion": "client.authentication.k8s.io/v1beta1",\n                "kind": "ExecCredential"\n            }\n            ']
    for output in outputs:
        instance.communicate.return_value = (output, '')
        with self.assertRaises(ConfigException) as context:
            ep = ExecProvider(self.input_ok)
            ep.run()
        self.assertIn('exec: malformed response. missing key', context.exception.args[0])