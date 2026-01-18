import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def test_multi_disconnected_iterable(tmpdir):
    metawf = pe.Workflow(name='meta')
    metawf.base_dir = tmpdir.strpath
    metawf.add_nodes([create_wf('wf%d' % i) for i in range(30)])
    eg = metawf.run(plugin='Linear')
    assert len(eg.nodes()) == 60