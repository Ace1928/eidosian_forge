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
def test_node_hash(tmpdir):
    from nipype.interfaces.utility import Function
    tmpdir.chdir()
    config.set_default_config()
    config.set('execution', 'stop_on_first_crash', True)
    config.set('execution', 'crashdump_dir', os.getcwd())

    def func1():
        return 1

    def func2(a):
        return a + 1
    n1 = pe.Node(Function(input_names=[], output_names=['a'], function=func1), name='n1')
    n2 = pe.Node(Function(input_names=['a'], output_names=['b'], function=func2), name='n2')
    w1 = pe.Workflow(name='test')

    def modify(x):
        return x + 1
    n1.inputs.a = 1
    w1.connect(n1, ('a', modify), n2, 'a')
    w1.base_dir = os.getcwd()
    from nipype.pipeline.plugins.base import DistributedPluginBase

    class EngineTestException(Exception):
        pass

    class RaiseError(DistributedPluginBase):

        def _submit_job(self, node, updatehash=False):
            raise EngineTestException('Submit called - cached=%s, updated=%s' % node.is_cached())
    with pytest.raises(EngineTestException) as excinfo:
        w1.run(plugin=RaiseError())
    assert str(excinfo.value).startswith('Submit called')
    w1.run(plugin='Linear')
    config.set('execution', 'local_hash_check', False)
    w1.run(plugin='Linear')
    config.set('execution', 'local_hash_check', True)
    w1 = pe.Workflow(name='test')
    w1.connect(n1, ('a', modify), n2, 'a')
    w1.base_dir = os.getcwd()
    w1.run(plugin=RaiseError())