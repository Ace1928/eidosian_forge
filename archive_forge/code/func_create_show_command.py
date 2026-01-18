import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import floatingips
def create_show_command(self, list_value, get_value):
    mock_floatingip_manager = mock.Mock()
    mock_floatingip_manager.list.return_value = list_value
    mock_floatingip_manager.get.return_value = get_value
    mock_client = mock.Mock()
    mock_client.floatingip = mock_floatingip_manager
    blazar_shell = shell.BlazarShell()
    blazar_shell.client = mock_client
    return (floatingips.ShowFloatingIP(blazar_shell, mock.Mock()), mock_floatingip_manager)