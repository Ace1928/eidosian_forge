from __future__ import (absolute_import, division, print_function)
import testinfra.utils.ansible_runner
def test_is_not_conjurized(host):
    identity_file = host.file('/etc/conjur.identity')
    assert not identity_file.exists
    conf_file = host.file('/etc/conjur.conf')
    assert not conf_file.exists