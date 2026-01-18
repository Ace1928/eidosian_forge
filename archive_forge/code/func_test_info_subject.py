import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_info_subject():
    assert subj_1.exists()
    expected_output = f'<Subject Object> BBRCDEV_S02627 `001` (project: pyxnat_tests) (Gender: U) 1 experiment {central._server}/data/projects/pyxnat_tests/subjects/BBRCDEV_S02627?format=html'
    assert str(subj_1) == expected_output