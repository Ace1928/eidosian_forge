import os
import testinfra.utils.ansible_runner
def test_mongod_port(host):
    port = include_vars(host)['ansible_facts']['config_port']
    s = host.socket('tcp://0.0.0.0:{0}'.format(port))
    assert s.is_listening