import argparse
from datetime import datetime
from unittest import mock
from blazarclient import exception
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import leases
def test_args2body_instance_reservation_params(self):
    args = argparse.Namespace(name=None, prolong_for=None, reduce_by=None, end_date=None, defer_by=None, advance_by=None, start_date=None, reservation=['id=798379a6-194c-45dc-ba34-1b5171d5552f,vcpus=3,memory_mb=1024,disk_gb=20,amount=4,affinity=False'])
    expected = {'reservations': [{'id': '798379a6-194c-45dc-ba34-1b5171d5552f', 'vcpus': 3, 'memory_mb': 1024, 'disk_gb': 20, 'amount': 4, 'affinity': 'False'}]}
    self.assertDictEqual(self.cl.args2body(args), expected)