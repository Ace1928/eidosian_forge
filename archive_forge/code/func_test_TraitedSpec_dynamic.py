import os
import warnings
import pytest
from ....utils.filemanip import split_filename
from ... import base as nib
from ...base import traits, Undefined
from ....interfaces import fsl
from ...utility.wrappers import Function
from ....pipeline import Node
from ..specs import get_filecopy_info
@pytest.mark.skip
def test_TraitedSpec_dynamic():
    from pickle import dumps, loads
    a = nib.BaseTraitedSpec()
    a.add_trait('foo', nib.traits.Int)
    a.foo = 1
    assign_a = lambda: setattr(a, 'foo', 'a')
    with pytest.raises(Exception):
        assign_a
    pkld_a = dumps(a)
    unpkld_a = loads(pkld_a)
    assign_a_again = lambda: setattr(unpkld_a, 'foo', 'a')
    with pytest.raises(Exception):
        assign_a_again