import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_model_set_reference():
    parent = create_model('parent')
    child = create_model('child')
    grandchild = create_model('child')
    parent.layers.append(child)
    assert parent.ref_names == tuple()
    parent.set_ref('kid', child)
    assert parent.ref_names == ('kid',)
    assert parent.get_ref('kid') is child
    child.layers.append(grandchild)
    with pytest.raises(KeyError):
        parent.get_ref('grandkid')
    parent.set_ref('grandkid', grandchild)
    assert parent.get_ref('grandkid') is grandchild
    parent.remove_node(grandchild)
    assert grandchild not in child.layers
    assert not parent.has_ref('grandkind')