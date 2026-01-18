import pytest
from ....testing import example_data
from ...niftyreg import get_custom_path
from ...niftyreg.tests.test_regutils import no_nifty_tool
from .. import UnaryStats, BinaryStats
@pytest.mark.skipif(no_nifty_tool(cmd='seg_stats'), reason='niftyseg is not installed')
def test_binary_stats():
    """Test for the seg_stats interfaces"""
    binarys = BinaryStats()
    cmd = get_custom_path('seg_stats', env_dir='NIFTYSEGDIR')
    assert binarys.cmd == cmd
    with pytest.raises(ValueError):
        binarys.run()
    in_file = example_data('im1.nii')
    binarys.inputs.in_file = in_file
    binarys.inputs.operand_value = 2
    binarys.inputs.operation = 'sa'
    expected_cmd = '{cmd} {in_file} -sa 2.00000000'.format(cmd=cmd, in_file=in_file)
    assert binarys.cmdline == expected_cmd