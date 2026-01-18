import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_cgroupsSuggestsRealSystem(self) -> None:
    """
        If the platform is Linux, and the cgroups file (faked out here) exists,
        and none of the paths starts with C{/docker/}, C{isDocker()} returns
        C{False}.
        """
    cgroupsFile = self.mktemp()
    with open(cgroupsFile, 'wb') as f:
        f.write(b'9:perf_event:/\n8:blkio:/\n7:net_cls:/\n6:freezer:/\n5:devices:/\n4:memory:/\n3:cpuacct,cpu:/\n2:cpuset:/\n1:name=systemd:/system')
    platform = Platform(None, 'linux')
    self.assertFalse(platform.isDocker(_initCGroupLocation=cgroupsFile))