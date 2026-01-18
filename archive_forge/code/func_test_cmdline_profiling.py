import os
import pytest
from .... import config
from ....utils.profiler import _use_resources
from ...base import traits, CommandLine, CommandLineInputSpec
from ... import utility as niu
@pytest.mark.skip(reason='inconsistent readings')
@pytest.mark.skipif(os.getenv('CI_SKIP_TEST', False), reason='disabled in CI tests')
@pytest.mark.parametrize('mem_gb,n_procs', [(0.5, 3), (2.2, 8), (0.8, 4), (1.5, 1)])
def test_cmdline_profiling(tmpdir, mem_gb, n_procs, use_resource_monitor):
    """
    Test runtime profiler correctly records workflow RAM/CPUs consumption
    of a CommandLine-derived interface
    """
    from nipype import config
    config.set('monitoring', 'sample_frequency', '0.2')
    tmpdir.chdir()
    iface = UseResources(mem_gb=mem_gb, n_procs=n_procs)
    result = iface.run()
    assert abs(mem_gb - result.runtime.mem_peak_gb) < 0.3, 'estimated memory error above .3GB'
    assert int(result.runtime.cpu_percent / 100 + 0.2) == n_procs, 'wrong number of threads estimated'