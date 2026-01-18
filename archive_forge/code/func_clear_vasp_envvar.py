import os
import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp
@pytest.fixture
def clear_vasp_envvar(monkeypatch):
    """Clear the environment variables which can be used to launch
    a VASP calculation."""
    for envvar in Vasp.env_commands:
        monkeypatch.delenv(envvar, raising=False)
        assert envvar not in os.environ
    yield