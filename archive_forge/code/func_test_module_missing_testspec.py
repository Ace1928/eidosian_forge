import pytest
from cirq.testing.json import spec_for
def test_module_missing_testspec():
    with pytest.raises(ValueError, match='TestSpec'):
        spec_for('cirq.testing.test_data.test_module_missing_testspec')