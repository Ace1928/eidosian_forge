import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from .. import UnaryMaths, BinaryMaths, BinaryMathsInteger, TupleMaths, Merge
@pytest.mark.skipif(no_nifty_tool(cmd='seg_maths'), reason='niftyseg is not installed')
def test_int_binary_maths():
    ibinarym = BinaryMathsInteger()
    cmd = get_custom_path('seg_maths', env_dir='NIFTYSEGDIR')
    assert ibinarym.cmd == cmd
    with pytest.raises(ValueError):
        ibinarym.run()
    in_file = example_data('im1.nii')
    ibinarym.inputs.in_file = in_file
    ibinarym.inputs.operand_value = 2
    ibinarym.inputs.operation = 'dil'
    ibinarym.inputs.output_datatype = 'float'
    expected_cmd = '{cmd} {in_file} -dil 2 -odt float {out_file}'.format(cmd=cmd, in_file=in_file, out_file='im1_dil.nii')
    assert ibinarym.cmdline == expected_cmd