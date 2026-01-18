import os
import testinfra.utils.ansible_runner
def test_mongod_replicaset(host):
    """
    Ensure that the MongoDB config replicaset has been created successfully
    """
    port = include_vars(host)['ansible_facts']['config_port']
    cmd = "mongosh --port {0} --eval 'rs.status()'".format(port)
    if host.ansible.get_variables()['inventory_hostname'] == 'fedora':
        r = host.run(cmd)
        assert 'cfg' in r.stdout
        assert 'almalinux_8:{0}'.format(port) in r.stdout
        assert 'fedora:{0}'.format(port) in r.stdout
        assert 'ubuntu_22_04:{0}'.format(port) in r.stdout
        assert 'ubuntu_22:{0}'.format(port) in r.stdout
        assert 'debian_bullseye:{0}'.format(port) in r.stdout