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
def test_ImageFile():
    x = nib.BaseInterface().inputs
    x.add_trait('nifti', nib.ImageFile(types=['nifti1', 'dicom']))
    x.add_trait('anytype', nib.ImageFile())
    with pytest.raises(ValueError):
        x.add_trait('newtype', nib.ImageFile(types=['nifti10']))
    x.add_trait('nocompress', nib.ImageFile(types=['mgh'], allow_compressed=False))
    with pytest.raises(nib.TraitError):
        x.nifti = 'test.mgz'
    x.nifti = 'test.nii'
    x.anytype = 'test.xml'
    with pytest.raises(nib.TraitError):
        x.nocompress = 'test.mgz'
    x.nocompress = 'test.mgh'