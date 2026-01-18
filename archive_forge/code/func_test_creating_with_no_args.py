import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
def test_creating_with_no_args(self):
    ret = loading.register_auth_argparse_arguments(self.p, [])
    self.assertIsNone(ret)
    self.assertIn('--os-auth-type', self.p.format_usage())