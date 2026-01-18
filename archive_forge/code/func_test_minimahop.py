from ase import Atoms, Atom
from ase.build import fcc111
from ase.optimize.minimahopping import MinimaHopping
from ase.constraints import FixAtoms
def test_minimahop(asap3, testdir):
    atoms = fcc111('Pt', (2, 2, 1), vacuum=7.0, orthogonal=True)
    adsorbate = Atoms([Atom('Cu', atoms[2].position + (0.0, 0.0, 2.5)), Atom('Cu', atoms[2].position + (0.0, 0.0, 5.0))])
    atoms.extend(adsorbate)
    constraints = [FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Pt'])]
    atoms.set_constraint(constraints)
    calc = asap3.EMT()
    atoms.calc = calc
    hop = MinimaHopping(atoms, Ediff0=2.5, T0=2000.0, beta1=1.2, beta2=1.2, mdmin=1)
    hop(totalsteps=3)
    hop(maxtemp=3000)