import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_info_resource():
    assert resource_1.exists()
    expected_output = f'<Resource Object> 19551 `obscure_algorithm_output` (66 files 2.06 GB)'
    assert str(resource_1) == expected_output