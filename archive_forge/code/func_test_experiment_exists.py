import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_experiment_exists():
    assert exp_1.exists()
    assert isinstance(exp_1, object)
    assert isinstance(exp_1, pyxnat.core.resources.Experiment)
    assert str(exp_1) != '<Experiment Object> GHJ'