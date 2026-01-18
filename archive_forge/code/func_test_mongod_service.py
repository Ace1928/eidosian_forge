import os
import testinfra.utils.ansible_runner
def test_mongod_service(host):
    mongod_service = include_vars(host)['ansible_facts']['mongod_service']
    s = host.service(mongod_service)
    assert s.is_running
    assert s.is_enabled