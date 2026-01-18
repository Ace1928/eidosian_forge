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
def test_node_get_output():
    mod1 = pe.Node(interface=EngineTestInterface(), name='mod1')
    mod1.inputs.input1 = 1
    mod1.run()
    assert mod1.get_output('output1') == [1, 1]
    mod1._result = None
    assert mod1.get_output('output1') == [1, 1]