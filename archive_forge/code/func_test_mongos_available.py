import os
import testinfra.utils.ansible_runner
def test_mongos_available(host):
    cmd = host.run('mongos --version')
    assert cmd.rc == 0
    assert 'mongos version' in cmd.stdout