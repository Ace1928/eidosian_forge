import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import hosts
def create_update_command(self, list_value):
    mock_host_manager = mock.Mock()
    mock_host_manager.list.return_value = list_value
    mock_client = mock.Mock()
    mock_client.host = mock_host_manager
    blazar_shell = shell.BlazarShell()
    blazar_shell.client = mock_client
    return (hosts.UpdateHost(blazar_shell, mock.Mock()), mock_host_manager)