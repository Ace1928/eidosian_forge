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
def test_TraitedSpec_withFile(setup_file):
    tmp_infile = setup_file
    tmpd, nme = os.path.split(tmp_infile)
    assert os.path.exists(tmp_infile)

    class spec2(nib.TraitedSpec):
        moo = nib.File(exists=True)
        doo = nib.traits.List(nib.File(exists=True))
    infields = spec2(moo=tmp_infile, doo=[tmp_infile])
    hashval = infields.get_hashval(hash_method='content')
    assert hashval[1] == 'a00e9ee24f5bfa9545a515b7a759886b'