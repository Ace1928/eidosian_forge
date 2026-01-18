from uuid import uuid1
from pyxnat import Interface
from pyxnat import jsonutil
import os.path as op
def test_inspect_resources():
    assert 'xnat_E02579' in central.inspect.experiment_values('xnat:mrSessionData', 'cs_schizbull08')
    assert 'xnat_E02580' in central.inspect.assessor_values('xnat:mrSessionData', 'cs_schizbull08')
    assert 'anat' in central.inspect.scan_values('xnat:mrSessionData', 'cs_schizbull08')
    assert isinstance(central.inspect.experiment_types(), list)
    assert isinstance(central.inspect.assessor_types(), list)
    assert isinstance(central.inspect.scan_types(), list)
    assert isinstance(central.inspect.reconstruction_types(), list)
    assert isinstance(central.inspect.project_values(), list)
    assert isinstance(central.inspect.subject_values(), list)