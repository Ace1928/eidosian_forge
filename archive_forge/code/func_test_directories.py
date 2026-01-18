from __future__ import (absolute_import, division, print_function)
import os
import testinfra.utils.ansible_runner
def test_directories(host):
    dirs = ['/etc/grafana', '/var/log/grafana', '/var/lib/grafana', '/var/lib/grafana/dashboards', '/var/lib/grafana/plugins']
    files = ['/etc/grafana/grafana.ini']
    for directory in dirs:
        d = host.file(directory)
        assert d.is_directory
        assert d.exists
    for file in files:
        f = host.file(file)
        assert f.exists
        assert f.is_file