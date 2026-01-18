import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
@utils.mock_plugin()
def test_env_overrides_default_opt(self, m):
    name = uuid.uuid4().hex
    val = uuid.uuid4().hex
    self.env('OS_A_STR', val)
    klass = loading.register_auth_argparse_arguments(self.p, [], default=name)
    self.assertIsInstance(klass, utils.MockLoader)
    opts = self.p.parse_args([])
    a = loading.load_auth_from_argparse_arguments(opts)
    self.assertEqual(val, a['a_str'])