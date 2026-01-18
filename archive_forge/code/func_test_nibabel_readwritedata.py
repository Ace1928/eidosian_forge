import io
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from packaging.version import Version
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.cifti2.parse_cifti2 import _Cifti2AsNiftiHeader
from nibabel.tests import test_nifti2 as tn2
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from nibabel.tmpdirs import InTemporaryDirectory
@needs_nibabel_data('nitest-cifti2')
def test_nibabel_readwritedata():
    with InTemporaryDirectory():
        for name in datafiles:
            img = nib.load(name)
            nib.save(img, 'test.nii')
            img2 = nib.load('test.nii')
            assert len(img.header.matrix) == len(img2.header.matrix)
            for mim1, mim2 in zip(img.header.matrix, img2.header.matrix):
                named_maps1 = [m_ for m_ in mim1 if isinstance(m_, ci.Cifti2NamedMap)]
                named_maps2 = [m_ for m_ in mim2 if isinstance(m_, ci.Cifti2NamedMap)]
                assert len(named_maps1) == len(named_maps2)
                for map1, map2 in zip(named_maps1, named_maps2):
                    assert map1.map_name == map2.map_name
                    if map1.label_table is None:
                        assert map2.label_table is None
                    else:
                        assert len(map1.label_table) == len(map2.label_table)
            assert_array_almost_equal(img.dataobj, img2.dataobj)