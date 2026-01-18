import os
import struct
import unittest
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from nibabel import nifti1 as nifti1
from nibabel.affines import from_matvec
from nibabel.casting import have_binary128, type_info
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import (
from nibabel.optpkg import optional_package
from nibabel.pkg_info import cmp_pkg_version
from nibabel.spatialimages import HeaderDataError
from nibabel.tmpdirs import InTemporaryDirectory
from ..freesurfer import load as mghload
from ..orientations import aff2axcodes
from ..testing import (
from . import test_analyze as tana
from . import test_spm99analyze as tspm
from .nibabel_data import get_nibabel_data, needs_nibabel_data
from .test_arraywriters import IUINT_TYPES, rt_err_estimate
from .test_orientations import ALL_ORNTS
@dicom_test
def test_nifti_dicom_extension():
    nim = load(image_file)
    hdr = nim.header
    exts_container = hdr.extensions
    dcmext = Nifti1DicomExtension(2, b'')
    assert dcmext.get_content().__class__ == pydicom.dataset.Dataset
    assert len(dcmext.get_content().values()) == 0
    dcmext = Nifti1DicomExtension(2, None)
    assert dcmext.get_content().__class__ == pydicom.dataset.Dataset
    assert len(dcmext.get_content().values()) == 0
    ds = pydicom.dataset.Dataset()
    ds.add_new((16, 32), 'LO', 'NiPy')
    dcmext = Nifti1DicomExtension(2, ds)
    assert dcmext.get_content().__class__ == pydicom.dataset.Dataset
    assert len(dcmext.get_content().values()) == 1
    assert dcmext.get_content().PatientID == 'NiPy'
    dcmbytes_explicit = struct.pack('<HH2sH4s', 16, 32, b'LO', 4, b'NiPy')
    dcmext = Nifti1DicomExtension(2, dcmbytes_explicit)
    assert dcmext.__class__ == Nifti1DicomExtension
    assert dcmext._guess_implicit_VR() is False
    assert dcmext._is_little_endian is True
    assert dcmext.get_code() == 2
    assert dcmext.get_content().PatientID == 'NiPy'
    assert len(dcmext.get_content().values()) == 1
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_explicit
    assert dcmext.get_sizeondisk() % 16 == 0
    dcmbytes_implicit = struct.pack('<HHL4s', 16, 32, 4, b'NiPy')
    dcmext = Nifti1DicomExtension(2, dcmbytes_implicit)
    assert dcmext._guess_implicit_VR() is True
    assert dcmext.get_code() == 2
    assert dcmext.get_content().PatientID == 'NiPy'
    assert len(dcmext.get_content().values()) == 1
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_implicit
    assert dcmext.get_sizeondisk() % 16 == 0
    dcmbytes_explicit_be = struct.pack('>2H2sH4s', 16, 32, b'LO', 4, b'NiPy')
    hdr_be = Nifti1Header(endianness='>')
    dcmext = Nifti1DicomExtension(2, dcmbytes_explicit_be, parent_hdr=hdr_be)
    assert dcmext.__class__ == Nifti1DicomExtension
    assert dcmext._guess_implicit_VR() is False
    assert dcmext.get_code() == 2
    assert dcmext.get_content().PatientID == 'NiPy'
    assert dcmext.get_content()[16, 32].value == 'NiPy'
    assert len(dcmext.get_content().values()) == 1
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_explicit_be
    assert dcmext.get_sizeondisk() % 16 == 0
    dcmext = Nifti1DicomExtension(2, ds, parent_hdr=hdr_be)
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_explicit_be
    assert exts_container.count('dicom') == 0
    exts_container.append(dcmext)
    assert exts_container.count('dicom') == 1
    assert exts_container.get_codes() == [6, 6, 2]
    assert dcmext._mangle(dcmext.get_content()) == dcmbytes_explicit_be
    assert dcmext.get_sizeondisk() % 16 == 0
    with pytest.raises(TypeError):
        Nifti1DicomExtension(2, 0)