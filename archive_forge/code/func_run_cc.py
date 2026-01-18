import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def run_cc(self, ccinterface, expected_components, expected_header='CompCor', expected_n_components=None, expected_metadata=None):
    ccresult = ccinterface.run()
    expected_file = ccinterface._list_outputs()['components_file']
    assert ccresult.outputs.components_file == expected_file
    assert os.path.exists(expected_file)
    assert os.path.getsize(expected_file) > 0
    with open(ccresult.outputs.components_file, 'r') as components_file:
        header = components_file.readline().rstrip().split('\t')
        components_data = np.loadtxt(components_file, delimiter='\t')
    if expected_n_components is None:
        expected_n_components = min(ccinterface.inputs.num_components, self.fake_data.shape[3])
    assert header == [f'{expected_header}{i:02d}' for i in range(expected_n_components)]
    assert components_data.shape == (self.fake_data.shape[3], expected_n_components)
    assert close_up_to_column_sign(components_data[:, :2], expected_components)
    if ccinterface.inputs.save_metadata:
        expected_metadata_file = ccinterface._list_outputs()['metadata_file']
        assert ccresult.outputs.metadata_file == expected_metadata_file
        assert os.path.exists(expected_metadata_file)
        assert os.path.getsize(expected_metadata_file) > 0
        with open(ccresult.outputs.metadata_file, 'r') as metadata_file:
            components_metadata = [line.rstrip().split('\t') for line in metadata_file]
            components_metadata = {i: j for i, j in zip(components_metadata[0], components_metadata[1])}
            assert components_metadata == expected_metadata
    return ccresult