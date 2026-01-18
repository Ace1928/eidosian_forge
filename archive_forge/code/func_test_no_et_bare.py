import os
from .. import get_info
from ..info import get_nipype_gitversion
import pytest
def test_no_et_bare(tmp_path):
    from unittest.mock import patch
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces.base import BaseInterface
    et = os.getenv('NIPYPE_NO_ET') is None
    with patch.object(BaseInterface, '_etelemetry_version_data', {}):
        f = niu.Function(function=_check_no_et)
        res = f.run()
        assert res.outputs.out == et
        n = pe.Node(niu.Function(function=_check_no_et), name='n', base_dir=str(tmp_path))
        res = n.run()
        assert res.outputs.out == et
        wf1 = pe.Workflow(name='wf1', base_dir=str(tmp_path))
        wf1.add_nodes([pe.Node(niu.Function(function=_check_no_et), name='n')])
        res = wf1.run()
        assert next(iter(res.nodes)).result.outputs.out == et