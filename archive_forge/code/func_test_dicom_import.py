import os
import pytest
from nipype.testing import example_data
import nipype.interfaces.spm.utils as spmu
from nipype.interfaces.base import isdefined
from nipype.utils.filemanip import split_filename, fname_presuffix
from nipype.interfaces.base import TraitError
def test_dicom_import():
    dicom = example_data(infile='dicomdir/123456-1-1.dcm')
    di = spmu.DicomImport(matlab_cmd='mymatlab')
    assert di.inputs.matlab_cmd == 'mymatlab'
    assert di.inputs.output_dir_struct == 'flat'
    assert di.inputs.output_dir == './converted_dicom'
    assert di.inputs.format == 'nii'
    assert not di.inputs.icedims
    with pytest.raises(TraitError):
        di.inputs.trait_set(output_dir_struct='wrong')
    with pytest.raises(TraitError):
        di.inputs.trait_set(format='FAT')
    with pytest.raises(TraitError):
        di.inputs.trait_set(in_files=['does_sfd_not_32fn_exist.dcm'])
    di.inputs.in_files = [dicom]
    assert di.inputs.in_files == [dicom]