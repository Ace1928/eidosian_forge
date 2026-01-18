import argparse
from datetime import datetime
from unittest import mock
from blazarclient import exception
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import leases
def test_args2body_host_reservation_params(self):
    args = argparse.Namespace(name=None, prolong_for=None, reduce_by=None, end_date=None, defer_by=None, advance_by=None, start_date=None, reservation=['id=798379a6-194c-45dc-ba34-1b5171d5552f,max=3,hypervisor_properties=["and", [">=", "$vcpus", "4"], [">=", "$memory_mb", "8192"]],resource_properties=["==", "$extra_key", "extra_value"]'])
    expected = {'reservations': [{'id': '798379a6-194c-45dc-ba34-1b5171d5552f', 'max': 3, 'hypervisor_properties': '["and", [">=", "$vcpus", "4"], [">=", "$memory_mb", "8192"]]', 'resource_properties': '["==", "$extra_key", "extra_value"]'}]}
    self.assertDictEqual(self.cl.args2body(args), expected)