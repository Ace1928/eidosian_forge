import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
def test_load_with_nothing(self):
    loading.register_auth_argparse_arguments(self.p, [])
    opts = self.p.parse_args([])
    self.assertIsNone(loading.load_auth_from_argparse_arguments(opts))