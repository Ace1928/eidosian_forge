import os
import testinfra.utils.ansible_runner
def test_mongo_available(host):
    cmd = host.run('mongosh --version')
    assert cmd.rc == 0