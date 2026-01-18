import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ..dwi import FitDwi, DwiTool
from ...niftyreg.tests.test_regutils import no_nifty_tool
@pytest.mark.skipif(no_nifty_tool(cmd='dwi_tool'), reason='niftyfit is not installed')
def test_dwi_tool():
    """Testing DwiTool interface."""
    dwi_tool = DwiTool()
    cmd = get_custom_path('dwi_tool', env_dir='NIFTYFITDIR')
    assert dwi_tool.cmd == cmd
    with pytest.raises(ValueError):
        dwi_tool.run()
    in_file = example_data('dwi.nii.gz')
    bval_file = example_data('bvals')
    bvec_file = example_data('bvecs')
    b0_file = example_data('b0.nii')
    mask_file = example_data('mask.nii.gz')
    dwi_tool.inputs.source_file = in_file
    dwi_tool.inputs.mask_file = mask_file
    dwi_tool.inputs.bval_file = bval_file
    dwi_tool.inputs.bvec_file = bvec_file
    dwi_tool.inputs.b0_file = b0_file
    dwi_tool.inputs.dti_flag = True
    cmd_tmp = '{cmd} -source {in_file} -bval {bval} -bvec {bvec} -b0 {b0} -mask {mask} -dti -famap {fa} -logdti2 {log} -mcmap {mc} -mdmap {md} -rgbmap {rgb} -syn {syn} -v1map {v1}'
    expected_cmd = cmd_tmp.format(cmd=cmd, in_file=in_file, bval=bval_file, bvec=bvec_file, b0=b0_file, mask=mask_file, fa='dwi_famap.nii.gz', log='dwi_logdti2.nii.gz', mc='dwi_mcmap.nii.gz', md='dwi_mdmap.nii.gz', rgb='dwi_rgbmap.nii.gz', syn='dwi_syn.nii.gz', v1='dwi_v1map.nii.gz')
    assert dwi_tool.cmdline == expected_cmd