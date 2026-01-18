import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_info_experiment():
    assert exp_1.exists()
    expected_output = f'<Experiment Object> BBRCDEV_E03106 `001_obscured` (subject: BBRCDEV_S02627 `001`) (project: pyxnat_tests) 4 scans 1 resource (created on 2024-01-22 10:25:48.637) {central._server}/data/projects/pyxnat_tests/subjects/BBRCDEV_S02627/experiments/BBRCDEV_E03106?format=html'
    assert str(exp_1) == expected_output