import os
import yaml
import testinfra.utils.ansible_runner
def test_mongo_shell_connectivity(host):
    """
    Tests that we can connect to mongos via the shell annd run a cmd
    """
    facts = include_vars(host)['ansible_facts']
    port = facts.get('mongod_port', 27017)
    user = facts.get('mongod_admin_user', 'admin')
    pwd = facts.get('mongodb_default_admin_pwd', 'admin')
    cmd = host.run("mongosh admin --username {user} --password {pwd} --port {port} --eval 'db.runCommand({{listDatabases: 1}})'".format(user=user, pwd=pwd, port=port))
    assert cmd.rc == 0
    assert 'admin' in cmd.stdout