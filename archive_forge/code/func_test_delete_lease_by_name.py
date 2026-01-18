import argparse
from datetime import datetime
from unittest import mock
from blazarclient import exception
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import leases
def test_delete_lease_by_name(self):
    delete_lease, lease_manager = self.create_delete_command()
    lease_manager.list.return_value = [{'id': FIRST_LEASE, 'name': 'first-lease'}, {'id': SECOND_LEASE, 'name': 'second-lease'}]
    lease_manager.delete.return_value = None
    args = argparse.Namespace(id='second-lease')
    delete_lease.run(args)
    lease_manager.list.assert_called_once_with()
    lease_manager.delete.assert_called_once_with(SECOND_LEASE)