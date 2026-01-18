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
@pytest.mark.parametrize('x_inp, f_exp', [(3, [6]), ([2, 3], [4, 6]), ((2, 3), [4, 6]), (range(3), [0, 2, 4]), ('Str', ['StrStr']), (['Str1', 'Str2'], ['Str1Str1', 'Str2Str2'])])
def test_mapnode_iterfield_type(x_inp, f_exp):
    from nipype import MapNode, Function

    def double_func(x):
        return 2 * x
    double = Function(['x'], ['f_x'], double_func)
    double_node = MapNode(double, name='double', iterfield=['x'])
    double_node.inputs.x = x_inp
    res = double_node.run()
    assert res.outputs.f_x == f_exp