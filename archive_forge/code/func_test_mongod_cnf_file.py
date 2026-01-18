import os
import testinfra.utils.ansible_runner
def test_mongod_cnf_file(host):
    mongodb_user = include_vars(host)['ansible_facts']['mongodb_user']
    mongodb_group = include_vars(host)['ansible_facts']['mongodb_group']
    f = host.file('/etc/mongod.conf')
    assert f.exists
    assert f.user == mongodb_user
    assert f.group == mongodb_group