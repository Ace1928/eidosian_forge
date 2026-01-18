from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_global_experiment_listing():
    assert central.array.experiments(project_id='ixi', experiment_type='xnat:mrSessionData')