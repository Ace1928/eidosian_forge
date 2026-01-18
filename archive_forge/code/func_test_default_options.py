import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
@utils.mock_plugin()
def test_default_options(self, m):
    name = uuid.uuid4().hex
    argv = ['--os-auth-type', name, '--os-a-float', str(self.a_float)]
    klass = loading.register_auth_argparse_arguments(self.p, argv)
    self.assertIsInstance(klass, utils.MockLoader)
    opts = self.p.parse_args(argv)
    self.assertEqual(name, opts.os_auth_type)
    a = loading.load_auth_from_argparse_arguments(opts)
    self.assertEqual(self.a_float, a['a_float'])
    self.assertEqual(3, a['a_int'])