import os
import testinfra.utils.ansible_runner
def test_mongos_shell_connectivity(host):
    """
    Tests that we can connect to mongos via the shell annd run a cmd
    """
    if host.ansible.get_variables()['inventory_hostname'] != 'config1':
        port = include_vars(host)['ansible_facts']['mongos_port']
        cmd = host.run("mongosh admin --username admin --password admin --port {0} --eval 'db.runCommand({{listDatabases: 1}})'".format(port))
        assert cmd.rc == 0
        assert 'config' in cmd.stdout
        assert 'admin' in cmd.stdout