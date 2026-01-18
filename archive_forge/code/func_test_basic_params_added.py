import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
@utils.mock_plugin()
def test_basic_params_added(self, m):
    name = uuid.uuid4().hex
    argv = ['--os-auth-plugin', name]
    ret = loading.register_auth_argparse_arguments(self.p, argv)
    self.assertIsInstance(ret, utils.MockLoader)
    for n in ('--os-a-int', '--os-a-bool', '--os-a-float'):
        self.assertIn(n, self.p.format_usage())
    m.assert_called_once_with(name)