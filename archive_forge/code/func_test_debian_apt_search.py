import os
import testinfra.utils.ansible_runner
def test_debian_apt_search(host):
    if host.system_info.distribution == 'debian' or host.system_info.distribution == 'ubuntu':
        cmd = host.run('apt search mongodb')
        assert cmd.rc == 0
        assert 'mongodb' in cmd.stdout