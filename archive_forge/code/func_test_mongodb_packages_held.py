import os
import testinfra.utils.ansible_runner
def test_mongodb_packages_held(host):
    test_apt = host.run('which apt-mark')
    if test_apt.rc == 0:
        c = 'apt-mark showhold'
    else:
        c = 'yum versionlock list'
    cmd = host.run(c)
    assert cmd.rc == 0
    assert 'mongodb-org' not in cmd.stdout