from numpy.linalg import norm
from ase.collections import s22
from ase.calculators.turbomole import Turbomole
def polarization_cycle(partition_1, partition_2, charges_2=None):
    """Performs an iteration of a polarization calculation."""
    properties = {}
    calc = Turbomole(atoms=partition_1, **params)
    if charges_2 is not None:
        calc.embed(charges=charges_2, positions=partition_2.positions)
    properties['e1'] = partition_1.get_potential_energy()
    properties['c1'] = partition_1.get_charges()
    calc = Turbomole(atoms=partition_2, **params)
    calc.embed(charges=properties['c1'], positions=partition_1.positions)
    properties['e2'] = partition_2.get_potential_energy()
    properties['c2'] = partition_2.get_charges()
    return properties