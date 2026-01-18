import os
import testinfra.utils.ansible_runner
def test_mongodb_packages_installed(host):
    p = host.package('mongodb-org')
    assert p.is_installed
    p = host.package('mongodb-org-server')
    assert p.is_installed
    p = host.package('mongodb-org-mongos')
    assert p.is_installed
    p = host.package('mongodb-org-tools')
    assert p.is_installed