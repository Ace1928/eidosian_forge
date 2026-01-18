import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from ..qt1 import FitQt1
@pytest.mark.skipif(no_nifty_tool(cmd='fit_qt1'), reason='niftyfit is not installed')
def test_fit_qt1():
    """Testing FitQt1 interface."""
    fit_qt1 = FitQt1()
    cmd = get_custom_path('fit_qt1', env_dir='NIFTYFITDIR')
    assert fit_qt1.cmd == cmd
    with pytest.raises(ValueError):
        fit_qt1.run()
    in_file = example_data('TI4D.nii.gz')
    fit_qt1.inputs.source_file = in_file
    cmd_tmp = '{cmd} -source {in_file} -comp {comp} -error {error} -m0map {map0} -mcmap {cmap} -res {res} -syn {syn} -t1map {t1map}'
    expected_cmd = cmd_tmp.format(cmd=cmd, in_file=in_file, comp='TI4D_comp.nii.gz', map0='TI4D_m0map.nii.gz', error='TI4D_error.nii.gz', cmap='TI4D_mcmap.nii.gz', res='TI4D_res.nii.gz', t1map='TI4D_t1map.nii.gz', syn='TI4D_syn.nii.gz')
    assert fit_qt1.cmdline == expected_cmd
    fit_qt1_2 = FitQt1(tis=[1, 2, 5], ir_flag=True)
    in_file = example_data('TI4D.nii.gz')
    fit_qt1_2.inputs.source_file = in_file
    cmd_tmp = '{cmd} -source {in_file} -IR -TIs 1.0 2.0 5.0 -comp {comp} -error {error} -m0map {map0} -mcmap {cmap} -res {res} -syn {syn} -t1map {t1map}'
    expected_cmd = cmd_tmp.format(cmd=cmd, in_file=in_file, comp='TI4D_comp.nii.gz', map0='TI4D_m0map.nii.gz', error='TI4D_error.nii.gz', cmap='TI4D_mcmap.nii.gz', res='TI4D_res.nii.gz', t1map='TI4D_t1map.nii.gz', syn='TI4D_syn.nii.gz')
    assert fit_qt1_2.cmdline == expected_cmd
    fit_qt1_3 = FitQt1(flips=[2, 4, 8], spgr=True)
    in_file = example_data('TI4D.nii.gz')
    fit_qt1_3.inputs.source_file = in_file
    cmd_tmp = '{cmd} -source {in_file} -comp {comp} -error {error} -flips 2.0 4.0 8.0 -m0map {map0} -mcmap {cmap} -res {res} -SPGR -syn {syn} -t1map {t1map}'
    expected_cmd = cmd_tmp.format(cmd=cmd, in_file=in_file, comp='TI4D_comp.nii.gz', map0='TI4D_m0map.nii.gz', error='TI4D_error.nii.gz', cmap='TI4D_mcmap.nii.gz', res='TI4D_res.nii.gz', t1map='TI4D_t1map.nii.gz', syn='TI4D_syn.nii.gz')
    assert fit_qt1_3.cmdline == expected_cmd