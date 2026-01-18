import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_experiment_not_exists():
    assert not exp_2.exists()
    assert isinstance(exp_2, object)
    assert isinstance(exp_2, pyxnat.core.resources.Experiment)
    assert str(exp_2) == '<Experiment Object> GHJ'