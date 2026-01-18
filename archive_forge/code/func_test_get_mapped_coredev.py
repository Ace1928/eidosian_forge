from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick.caches import opencas
from os_brick import exception
from os_brick.tests import base
@mock.patch('os_brick.executor.Executor._execute')
def test_get_mapped_coredev(self, moc_exec):
    out_ready = 'type  id  disk  status  write policy  device\n        cache  1  /dev/nvme0n1  Running  wt  -\n        └core  1  /dev/sdd      Active   -   /dev/cas1-1'
    err = ''
    engine = opencas.OpenCASEngine(root_helper=None, opencas_cache_id=1)
    moc_exec.return_value = (out_ready, err)
    ret1, ret2 = engine._get_mapped_coredev('/dev/cas1-1')
    self.assertEqual('1', ret1)
    self.assertEqual('/dev/sdd', ret2)