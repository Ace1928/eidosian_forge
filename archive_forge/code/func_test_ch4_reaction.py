import pytest
from ase.build import molecule
from ase.calculators.calculator import get_calculator_class
from ase.units import Ry
from ase.utils import workdir
@pytest.mark.calculator_lite
@calc('abinit', ecut=300, chksymbreak=0, toldfe=0.0001)
@calc('aims')
@calc('cp2k')
@calc('espresso', ecutwfc=300 / Ry)
@calc('gpaw', symmetry='off', mode='pw', txt='gpaw.txt', mixer={'beta': 0.6}, marks=[filterwarnings('ignore:.*?ignore_bad_restart_file'), filterwarnings('ignore:convert_string_to_fd')])
@calc('nwchem')
@calc('octopus', Spacing='0.4 * angstrom')
@calc('openmx')
@calc('siesta', marks=pytest.mark.xfail)
def test_ch4_reaction(factory):
    e_ch4 = _calculate(factory, 'CH4')
    e_c2h2 = _calculate(factory, 'C2H2')
    e_h2 = _calculate(factory, 'H2')
    energy = e_ch4 - 0.5 * e_c2h2 - 1.5 * e_h2
    print(energy)
    ref_energy = -2.8
    assert abs(energy - ref_energy) < 0.3