import os
import pytest
import shutil
from nipype.interfaces.dcm2nii import Dcm2niix
@pytest.mark.skipif(no_datalad, reason='Datalad required')
@pytest.mark.skipif(no_dcm2niix, reason='Dcm2niix required')
def test_dcm2niix_dti(fetch_data, tmpdir):
    tmpdir.chdir()
    datadir = tmpdir.mkdir('data').strpath
    dicoms = fetch_data(datadir, 'Siemens_Sag_DTI_20160825_145811')

    def assert_dti(res):
        """Some assertions we will make"""
        assert res.outputs.converted_files
        assert res.outputs.bvals
        assert res.outputs.bvecs
        outputs = [y for x, y in res.outputs.get().items()]
        if res.inputs.get('bids_format'):
            assert len(set(map(len, outputs))) == 1
        else:
            assert not res.outputs.bids
    dcm = Dcm2niix()
    dcm.inputs.source_dir = dicoms
    dcm.inputs.out_filename = '%u%z'
    assert_dti(dcm.run())
    outdir = tmpdir.mkdir('conversion').strpath
    dcm.inputs.output_dir = outdir
    dcm.inputs.bids_format = False
    assert_dti(dcm.run())