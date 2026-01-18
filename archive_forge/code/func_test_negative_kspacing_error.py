import os
import pytest
from ase.build import bulk
from .filecmp_ignore_whitespace import filecmp_ignore_whitespace
@calc('vasp')
def test_negative_kspacing_error(factory, write_kpoints):
    with pytest.raises(ValueError):
        write_kpoints(factory, kspacing=-0.5)