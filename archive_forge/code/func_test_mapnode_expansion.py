import os
from copy import deepcopy
import pytest
from .... import config
from ....interfaces import utility as niu
from ....interfaces import base as nib
from ... import engine as pe
from ..utils import merge_dict
from .test_base import EngineTestInterface
from .test_utils import UtilsTestInterface
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.spm as spm
import os
from io import StringIO
from nipype.utils.config import config
def test_mapnode_expansion(tmpdir):
    tmpdir.chdir()
    from nipype import MapNode, Function

    def func1(in1):
        return in1 + 1
    mapnode = MapNode(Function(function=func1), iterfield='in1', name='mapnode', n_procs=2, mem_gb=2)
    mapnode.inputs.in1 = [1, 2]
    for idx, node in mapnode._make_nodes():
        for attr in ('overwrite', 'run_without_submitting', 'plugin_args'):
            assert getattr(node, attr) == getattr(mapnode, attr)
        for attr in ('_n_procs', '_mem_gb'):
            assert getattr(node, attr) == getattr(mapnode, attr)