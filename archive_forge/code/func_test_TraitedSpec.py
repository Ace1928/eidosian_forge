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
def test_TraitedSpec():
    assert nib.TraitedSpec().get_hashval()
    assert nib.TraitedSpec().__repr__() == '\n\n'

    class spec(nib.TraitedSpec):
        foo = nib.traits.Int
        goo = nib.traits.Float(usedefault=True)
    assert spec().foo == Undefined
    assert spec().goo == 0.0
    specfunc = lambda x: spec(hoo=x)
    with pytest.raises(nib.traits.TraitError):
        specfunc(1)
    infields = spec(foo=1)
    hashval = ([('foo', 1), ('goo', '0.0000000000')], 'e89433b8c9141aa0fda2f8f4d662c047')
    assert infields.get_hashval() == hashval
    assert infields.__repr__() == '\nfoo = 1\ngoo = 0.0\n'