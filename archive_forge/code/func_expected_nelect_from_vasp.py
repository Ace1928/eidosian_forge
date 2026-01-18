import pytest
from ase.build import bulk
@pytest.fixture
def expected_nelect_from_vasp():
    return 12