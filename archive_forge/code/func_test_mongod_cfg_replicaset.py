import os
import testinfra.utils.ansible_runner
def test_mongod_cfg_replicaset(host):
    """
    Ensure that the MongoDB config replicaset has been created successfully
    """
    port = include_vars(host)['ansible_facts']['config_port']
    cmd = "mongo --port {0} --eval 'rs.status()'".format(port)
    if host.ansible.get_variables()['inventory_hostname'] == 'fedora':
        r = host.run(cmd)
        assert 'cfg' in r.stdout
        assert 'fedora.local:{0}'.format(port) in r.stdout
        assert 'ubuntu-18.local:{0}'.format(port) in r.stdout
        assert 'debian-buster.local:{0}'.format(port) in r.stdout
        assert 'debian-stretch.local:{0}'.format(port) in r.stdout
        assert 'centos-7.local:{0}'.format(port) in r.stdout