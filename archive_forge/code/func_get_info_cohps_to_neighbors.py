from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
def get_info_cohps_to_neighbors(self, path_to_cohpcar: str | None='COHPCAR.lobster', obj_cohpcar: CompleteCohp | None=None, isites: list[int] | None=None, only_bonds_to: list[str] | None=None, onlycation_isites: bool=True, per_bond: bool=True, summed_spin_channels: bool=False):
    """
        Return info about the cohps (coops or cobis) as a summed cohp object and a label
        from all sites mentioned in isites with neighbors.

        Args:
            path_to_cohpcar: str, path to COHPCAR or COOPCAR or COBICAR
            obj_cohpcar: CompleteCohp object
            isites: list of int that indicate the number of the site
            only_bonds_to: list of str, e.g. ["O"] to only show cohps of anything to oxygen
            onlycation_isites: if isites=None, only cation sites will be returned
            per_bond: will normalize per bond
            summed_spin_channels: will sum all spin channels

        Returns:
            str: label for cohp (str), CompleteCohp object which describes all cohps (coops or cobis)
                of the sites as given by isites and the other parameters
        """
    _summed_icohps, _list_icohps, _number_bonds, labels, atoms, final_isites = self.get_info_icohps_to_neighbors(isites=isites, onlycation_isites=onlycation_isites)
    with tempfile.TemporaryDirectory() as t:
        path = f'{t}/POSCAR.vasp'
        self.structure.to(filename=path, fmt='poscar')
        if not hasattr(self, 'completecohp'):
            if path_to_cohpcar is not None and obj_cohpcar is None:
                self.completecohp = CompleteCohp.from_file(fmt='LOBSTER', filename=path_to_cohpcar, structure_file=path, are_coops=self.are_coops, are_cobis=self.are_cobis)
            elif obj_cohpcar is not None:
                self.completecohp = obj_cohpcar
            else:
                raise ValueError('Please provide either path_to_cohpcar or obj_cohpcar')
    if len(self.Icohpcollection._list_atom1) != len(self.completecohp.bonds):
        raise ValueError('COHPCAR and ICOHPLIST do not fit together')
    is_spin_completecohp = Spin.down in self.completecohp.get_cohp_by_label('1').cohp
    if self.Icohpcollection.is_spin_polarized != is_spin_completecohp:
        raise ValueError('COHPCAR and ICOHPLIST do not fit together')
    if only_bonds_to is None:
        divisor = len(labels) if per_bond else 1
        plot_label = self._get_plot_label(atoms, per_bond)
        summed_cohp = self.completecohp.get_summed_cohp_by_label_list(label_list=labels, divisor=divisor, summed_spin_channels=summed_spin_channels)
    else:
        new_labels = []
        new_atoms = []
        for key, atompair, isite in zip(labels, atoms, final_isites):
            present = False
            for atomtype in only_bonds_to:
                if str(self.structure[isite].species.elements[0]) != atomtype:
                    if atomtype in (self._split_string(atompair[0])[0], self._split_string(atompair[1])[0]):
                        present = True
                elif atomtype == self._split_string(atompair[0])[0] and atomtype == self._split_string(atompair[1])[0]:
                    present = True
            if present:
                new_labels.append(key)
                new_atoms.append(atompair)
        if new_labels:
            divisor = len(new_labels) if per_bond else 1
            plot_label = self._get_plot_label(new_atoms, per_bond)
            summed_cohp = self.completecohp.get_summed_cohp_by_label_list(label_list=new_labels, divisor=divisor, summed_spin_channels=summed_spin_channels)
        else:
            plot_label = None
            summed_cohp = None
    return (plot_label, summed_cohp)