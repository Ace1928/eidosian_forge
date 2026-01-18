import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
def test_adapter_service_type(self):
    argv = ['--os-service-type', 'compute']
    loading.register_adapter_argparse_arguments(self.p, 'compute')
    opts = self.p.parse_args(argv)
    self.assertEqual('compute', opts.os_service_type)
    self.assertFalse(hasattr(opts, 'os_compute_service_type'))