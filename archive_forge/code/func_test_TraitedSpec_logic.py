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
def test_TraitedSpec_logic():

    class spec3(nib.TraitedSpec):
        _xor_inputs = ('foo', 'bar')
        foo = nib.traits.Int(xor=_xor_inputs, desc='foo or bar, not both')
        bar = nib.traits.Int(xor=_xor_inputs, desc='bar or foo, not both')
        kung = nib.traits.Float(requires=('foo',), position=0, desc='kung foo')

    class out3(nib.TraitedSpec):
        output = nib.traits.Int

    class MyInterface(nib.BaseInterface):
        input_spec = spec3
        output_spec = out3
    myif = MyInterface()
    myif.inputs.foo = 1
    assert myif.inputs.foo == 1
    set_bar = lambda: setattr(myif.inputs, 'bar', 1)
    with pytest.raises(IOError):
        set_bar()
    assert myif.inputs.foo == 1
    myif.inputs.kung = 2
    assert myif.inputs.kung == 2.0