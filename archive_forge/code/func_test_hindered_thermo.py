import numpy as np
from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.build import bulk
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.phonons import Phonons
from ase.thermochemistry import (IdealGasThermo, HarmonicThermo,
from ase.calculators.emt import EMT
def test_hindered_thermo():
    vibs = np.array([3049.06067, 3040.796863, 3001.661338, 2997.961647, 2866.153162, 2750.85546, 1436.792655, 1431.413595, 1415.952186, 1395.7263, 1358.412432, 1335.922737, 1167.009954, 1142.126116, 1013.91868, 803.400098, 783.026031, 310.448278, 136.112935, 112.939853, 103.926392, 77.262869, 60.278004, 25.825447])
    vib_energies = vibs / 8065.54429
    trans_barrier_energy = 0.049313
    rot_barrier_energy = 0.017675
    sitedensity = 1500000000000000.0
    rotationalminima = 6
    symmetrynumber = 1
    mass = 30.07
    inertia = 73.149
    thermo = HinderedThermo(vib_energies=vib_energies, trans_barrier_energy=trans_barrier_energy, rot_barrier_energy=rot_barrier_energy, sitedensity=sitedensity, rotationalminima=rotationalminima, symmetrynumber=symmetrynumber, mass=mass, inertia=inertia)
    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    target = 1.593
    assert helmholtz - target < 0.001