from __future__ import (absolute_import, division, print_function)
import os
import testinfra.utils.ansible_runner
def test_custom_auth_option(host):
    f = host.file('/etc/grafana/grafana.ini')
    assert f.contains('login_maximum_inactive_lifetime_days = 42')