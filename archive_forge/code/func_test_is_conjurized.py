from __future__ import (absolute_import, division, print_function)
import testinfra.utils.ansible_runner
def test_is_conjurized(host):
    identity_file = host.file('/etc/conjur.identity')
    assert identity_file.exists
    assert identity_file.user == 'root'
    conf_file = host.file('/etc/conjur.conf')
    assert conf_file.exists
    assert conf_file.user == 'root'