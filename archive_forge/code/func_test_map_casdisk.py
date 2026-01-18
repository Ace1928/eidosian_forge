from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick.caches import opencas
from os_brick import exception
from os_brick.tests import base
@mock.patch('os_brick.executor.Executor._execute')
@mock.patch('os_brick.caches.opencas.OpenCASEngine._get_mapped_casdev')
def test_map_casdisk(self, moc_get_mapped_casdev, moc_exec):
    engine = opencas.OpenCASEngine(root_helper=None, opencas_cache_id=1)
    moc_get_mapped_casdev.return_value = ''
    moc_exec.return_value = ('', '')
    engine._map_casdisk('/dev/sdd')
    moc_exec.assert_has_calls([mock.call('casadm', '-A', '-i', 1, '-d', '/dev/sdd', run_as_root=True, root_helper=None)])