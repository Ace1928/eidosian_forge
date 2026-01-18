import os
import testinfra.utils.ansible_runner
def test_mongodb_cgroup_module_installed(host):
    cmd = host.run('semodule --list-modules | grep mongodb_cgroup_memory')
    assert cmd.rc == 0