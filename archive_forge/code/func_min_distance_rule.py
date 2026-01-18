import os
import re
import numpy as np
from ase.units import eV, Ang
from ase.calculators.calculator import FileIOCalculator, ReadError
def min_distance_rule(self, sym1, sym2, ifcloselabel1=None, ifcloselabel2=None, elselabel1=None, max_distance=3.0):
    """Find pairs of atoms to label based on proximity.

        This is for, e.g., the ffsioh or catlow force field, where we
        would like to identify those O atoms that are close to H
        atoms.  For each H atoms, we must specially label one O atom.

        This function is a rule that allows to define atom labels (like O1,
        O2, O_H etc..)  starting from element symbols of an Atoms
        object that a force field can use and according to distance
        parameters.

        Example:
        atoms = read('some_xyz_format.xyz')
        a = Conditions(atoms)
        a.set_min_distance_rule('O', 'H', ifcloselabel1='O2',
                                ifcloselabel2='H', elselabel1='O1')
        new_atoms_labels = a.get_atom_labels()

        In the example oxygens O are going to be labeled as O2 if they
        are close to a hydrogen atom othewise are labeled O1.

        """
    if ifcloselabel1 is None:
        ifcloselabel1 = sym1
    if ifcloselabel2 is None:
        ifcloselabel2 = sym2
    if elselabel1 is None:
        elselabel1 = sym1
    self.atom_types.append([sym1, ifcloselabel1, elselabel1])
    self.atom_types.append([sym2, ifcloselabel2])
    dist_mat = self.atoms.get_all_distances()
    index_assigned_sym1 = []
    index_assigned_sym2 = []
    for i in range(len(self.atoms_symbols)):
        if self.atoms_symbols[i] == sym2:
            dist_12 = 1000
            index_assigned_sym2.append(i)
            for t in range(len(self.atoms_symbols)):
                if self.atoms_symbols[t] == sym1 and dist_mat[i, t] < dist_12 and (t not in index_assigned_sym1):
                    dist_12 = dist_mat[i, t]
                    closest_sym1_index = t
            index_assigned_sym1.append(closest_sym1_index)
    for i1, i2 in zip(index_assigned_sym1, index_assigned_sym2):
        if dist_mat[i1, i2] > max_distance:
            raise ValueError('Cannot unambiguously apply minimum-distance rule because pairings are not obvious.  If you wish to ignore this, then increase max_distance.')
    for s in range(len(self.atoms_symbols)):
        if s in index_assigned_sym1:
            self.atoms_labels[s] = ifcloselabel1
        elif s not in index_assigned_sym1 and self.atoms_symbols[s] == sym1:
            self.atoms_labels[s] = elselabel1
        elif s in index_assigned_sym2:
            self.atoms_labels[s] = ifcloselabel2