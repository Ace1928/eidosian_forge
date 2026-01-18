from glob import glob
from os.path import basename, dirname
from os.path import join as pjoin
from warnings import simplefilter
import numpy as np
import pytest
from numpy import array as npa
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import load as top_load
from .. import parrec
from ..fileholders import FileHolder
from ..nifti1 import Nifti1Extension, Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..parrec import (
from ..testing import assert_arr_dict_equal, clear_and_catch_warnings, suppress_warnings
from ..volumeutils import array_from_file
from . import test_spatialimages as tsi
from .test_arrayproxy import check_mmap
def test_exts2par():
    par_img = PARRECImage.from_filename(EG_PAR)
    nii_img = Nifti1Image.from_image(par_img)
    assert exts2pars(nii_img) == []
    assert exts2pars(nii_img.header) == []
    assert exts2pars(nii_img.header.extensions) == []
    assert exts2pars([]) == []
    with open(EG_PAR, 'rb') as fobj:
        hdr_dump = fobj.read()
        dump_ext = Nifti1Extension('comment', hdr_dump)
    nii_img.header.extensions.append(dump_ext)
    hdrs = exts2pars(nii_img)
    assert len(hdrs) == 1
    assert hdrs[0].get_slice_orientation() == 'transverse'
    nii_img.header.extensions.append(Nifti1Extension('comment', hdr_dump))
    hdrs = exts2pars(nii_img)
    assert len(hdrs) == 2
    assert hdrs[1].get_slice_orientation() == 'transverse'
    nii_img.header.extensions.append(Nifti1Extension('comment', b''))
    for source in (nii_img, nii_img.header, nii_img.header.extensions, list(nii_img.header.extensions)):
        hdrs = exts2pars(source)
        assert len(hdrs) == 2