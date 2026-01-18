import os
import testinfra.utils.ansible_runner
def test_mongod_config_custom_path(host):
    """
    Ensure that the custom path is respected
    """
    default_path = '/data/db'
    f = host.file(default_path)
    assert f.exists
    assert f.is_directory
    conf = host.file('/etc/mongod.conf').content_string
    assert 'dbPath: {0}'.format(default_path) in conf