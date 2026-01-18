import sys
import unittest.mock as mock
import pytest
import cirq_google as cg
from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook, QCSObjectsForNotebook
@mock.patch.dict('sys.modules', {'google.colab': mock.Mock()})
@mock.patch('cirq_google.engine.qcs_notebook.get_engine')
def test_get_qcs_objects_for_notebook_auth_fails(engine_mock):
    auth_mock = sys.modules['google.colab']
    auth_mock.auth.authenticate_user = mock.Mock(side_effect=Exception('mock auth failure'))
    fake_processor = cg.engine.SimulatedLocalProcessor(processor_id='tester', project_name='mock_project', device=cg.Sycamore)
    fake_engine = cg.engine.SimulatedLocalEngine([fake_processor])
    engine_mock.return_value = fake_engine
    result = get_qcs_objects_for_notebook()
    _assert_correct_types(result)
    assert not result.signed_in
    assert result.is_simulator
    assert result.project_id == 'fake_project'