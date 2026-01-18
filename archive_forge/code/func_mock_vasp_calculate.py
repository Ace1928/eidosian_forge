import os
import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp
@pytest.fixture
def mock_vasp_calculate(mocker):
    """Fixture which mocks the VASP run method, so a calculation cannot run.
    Acts as a safeguard for tests which want to test VASP,
    but avoid accidentally launching a calculation"""

    def _mock_run(self, command=None, out=None, directory=None):
        assert False, 'Test attempted to launch a calculation'
    mocker.patch('ase.calculators.vasp.Vasp._run', _mock_run)
    yield