import pytest
from winrm.protocol import Protocol
def test_get_command_output(protocol_fake):
    shell_id = protocol_fake.open_shell()
    command_id = protocol_fake.run_command(shell_id, 'ipconfig', ['/all'])
    std_out, std_err, status_code = protocol_fake.get_command_output(shell_id, command_id)
    assert status_code == 0
    assert b'Windows IP Configuration' in std_out
    assert len(std_err) == 0
    protocol_fake.cleanup_command(shell_id, command_id)
    protocol_fake.close_shell(shell_id)