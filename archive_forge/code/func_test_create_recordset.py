from unittest import mock
from osc_lib.tests import utils
from designateclient.tests.osc import resources
from designateclient.v2 import base
from designateclient.v2.cli import recordsets
def test_create_recordset(self):
    arg_list = ['6f106adb-0896-4114-b34f-4ac8dfee9465', 'example', '--type', 'A', '--record', '127.0.0.1', '--record', '127.0.0.2']
    verify_args = [('zone_id', '6f106adb-0896-4114-b34f-4ac8dfee9465'), ('name', 'example'), ('type', 'A'), ('record', ['127.0.0.1', '127.0.0.2'])]
    body = resources.load('recordset_create')
    self.dns_client.recordsets.create.return_value = body
    parsed_args = self.check_parser(self.cmd, arg_list, verify_args)
    columns, data = self.cmd.take_action(parsed_args)
    results = list(data)
    self.assertEqual(14, len(results))