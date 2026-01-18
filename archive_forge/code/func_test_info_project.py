import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_info_project():
    assert proj_1.exists()
    expected_output = f'<Project Object> pyxnat_tests `pyxnat tests` (private) 3 subjects 5 MR experiments 1 CT experiment 1 PET experiment (owner: {central._user}) (created on 2024-01-22 10:09:22.086) {central._server}/data/projects/pyxnat_tests?format=html'
    assert str(proj_1) == expected_output