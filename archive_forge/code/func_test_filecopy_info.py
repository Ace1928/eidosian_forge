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
def test_filecopy_info():

    class InputSpec(nib.TraitedSpec):
        foo = nib.traits.Int(desc='a random int')
        goo = nib.traits.Int(desc='a random int', mandatory=True)
        moo = nib.traits.Int(desc='a random int', mandatory=False)
        hoo = nib.traits.Int(desc='a random int', usedefault=True)
        zoo = nib.File(desc='a file', copyfile=False)
        woo = nib.File(desc='a file', copyfile=True)

    class DerivedInterface(nib.BaseInterface):
        input_spec = InputSpec
        resource_monitor = False

        def normalize_filenames(self):
            """A mock normalize_filenames for freesurfer interfaces that have one"""
            self.inputs.zoo = 'normalized_filename.ext'
    assert get_filecopy_info(nib.BaseInterface) == []
    info = get_filecopy_info(DerivedInterface)
    assert info[0]['key'] == 'woo'
    assert info[0]['copy']
    assert info[1]['key'] == 'zoo'
    assert not info[1]['copy']
    info = None
    derived = DerivedInterface()
    assert derived.inputs.zoo == Undefined
    info = get_filecopy_info(derived)
    assert derived.inputs.zoo == 'normalized_filename.ext'
    assert info[0]['key'] == 'woo'
    assert info[0]['copy']
    assert info[1]['key'] == 'zoo'
    assert not info[1]['copy']