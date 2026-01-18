import sys
import unittest.mock as mock
import pytest
import cirq_google as cg
from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook, QCSObjectsForNotebook
def test_get_qcs_objects_for_notebook_virtual():
    result = get_qcs_objects_for_notebook(virtual=True)
    _assert_correct_types(result)
    _assert_simulated_values(result)
    assert result.processor_id == 'rainbow'
    assert len(result.device.metadata.qubit_set) == 23
    result = get_qcs_objects_for_notebook(processor_id='weber', virtual=True)
    _assert_correct_types(result)
    _assert_simulated_values(result)
    assert result.processor_id == 'weber'
    assert len(result.device.metadata.qubit_set) == 53