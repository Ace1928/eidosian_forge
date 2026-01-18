import os
import testinfra.utils.ansible_runner
def test_mongodb_lock_file(host):
    f = host.file('/root/mongo_version_lock.success')
    assert not f.exists