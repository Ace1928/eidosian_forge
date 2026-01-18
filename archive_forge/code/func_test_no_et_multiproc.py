import os
from .. import get_info
from ..info import get_nipype_gitversion
import pytest
@pytest.mark.parametrize('plugin', ('MultiProc', 'LegacyMultiProc'))
@pytest.mark.parametrize('run_without_submitting', (True, False))
def test_no_et_multiproc(tmp_path, plugin, run_without_submitting):
    from unittest.mock import patch
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces.base import BaseInterface
    et = os.getenv('NIPYPE_NO_ET') is None
    expectation = et if run_without_submitting else False
    with patch.object(BaseInterface, '_etelemetry_version_data', {}):
        wf = pe.Workflow(name='wf2', base_dir=str(tmp_path))
        n = pe.Node(niu.Function(function=_check_no_et), run_without_submitting=run_without_submitting, name='n')
        wf.add_nodes([n])
        res = wf.run(plugin=plugin, plugin_args={'n_procs': 1})
        assert next(iter(res.nodes)).result.outputs.out is expectation