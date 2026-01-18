import os
import testinfra.utils.ansible_runner
def test_mongod_available(host):
    cmd = host.run('mongod --version')
    assert cmd.rc == 0
    assert 'db version' in cmd.stdout