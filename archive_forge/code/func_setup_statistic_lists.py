from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
def setup_statistic_lists(self):
    """Set up the statistics of environments for this LightStructureEnvironments."""
    self.statistics_dict = {'valences_origin': self.valences_origin, 'anion_list': {}, 'anion_number': None, 'anion_atom_list': {}, 'anion_atom_number': None, 'cation_list': {}, 'cation_number': None, 'cation_atom_list': {}, 'cation_atom_number': None, 'neutral_list': {}, 'neutral_number': None, 'neutral_atom_list': {}, 'neutral_atom_number': None, 'atom_coordination_environments_present': {}, 'ion_coordination_environments_present': {}, 'coordination_environments_ion_present': {}, 'coordination_environments_atom_present': {}, 'fraction_ion_coordination_environments_present': {}, 'fraction_atom_coordination_environments_present': {}, 'fraction_coordination_environments_ion_present': {}, 'fraction_coordination_environments_atom_present': {}, 'count_ion_present': {}, 'count_atom_present': {}, 'count_coordination_environments_present': {}}
    atom_stat = self.statistics_dict['atom_coordination_environments_present']
    ce_atom_stat = self.statistics_dict['coordination_environments_atom_present']
    fraction_atom_stat = self.statistics_dict['fraction_atom_coordination_environments_present']
    fraction_ce_atom_stat = self.statistics_dict['fraction_coordination_environments_atom_present']
    count_atoms = self.statistics_dict['count_atom_present']
    count_ce = self.statistics_dict['count_coordination_environments_present']
    for isite, site in enumerate(self.structure):
        site_species = []
        if self.valences != 'undefined':
            for sp, occ in site.species.items():
                valence = self.valences[isite]
                strspecie = str(Species(sp.symbol, valence))
                if valence < 0:
                    specie_list = self.statistics_dict['anion_list']
                    atomlist = self.statistics_dict['anion_atom_list']
                elif valence > 0:
                    specie_list = self.statistics_dict['cation_list']
                    atomlist = self.statistics_dict['cation_atom_list']
                else:
                    specie_list = self.statistics_dict['neutral_list']
                    atomlist = self.statistics_dict['neutral_atom_list']
                if strspecie not in specie_list:
                    specie_list[strspecie] = occ
                else:
                    specie_list[strspecie] += occ
                if sp.symbol not in atomlist:
                    atomlist[sp.symbol] = occ
                else:
                    atomlist[sp.symbol] += occ
                site_species.append((sp.symbol, valence, occ))
        if self.coordination_environments[isite] is not None:
            site_envs = [(ce_piece_dict['ce_symbol'], ce_piece_dict['ce_fraction']) for ce_piece_dict in self.coordination_environments[isite]]
            for ce_symbol, fraction in site_envs:
                if fraction is None:
                    continue
                count_ce.setdefault(ce_symbol, 0.0)
                count_ce[ce_symbol] += fraction
            for sp, occ in site.species.items():
                elmt = sp.symbol
                if elmt not in atom_stat:
                    atom_stat[elmt] = {}
                    count_atoms[elmt] = 0.0
                count_atoms[elmt] += occ
                for ce_symbol, fraction in site_envs:
                    if fraction is None:
                        continue
                    if ce_symbol not in atom_stat[elmt]:
                        atom_stat[elmt][ce_symbol] = 0.0
                    atom_stat[elmt][ce_symbol] += occ * fraction
                    if ce_symbol not in ce_atom_stat:
                        ce_atom_stat[ce_symbol] = {}
                    if elmt not in ce_atom_stat[ce_symbol]:
                        ce_atom_stat[ce_symbol][elmt] = 0.0
                    ce_atom_stat[ce_symbol][elmt] += occ * fraction
            if self.valences != 'undefined':
                ion_stat = self.statistics_dict['ion_coordination_environments_present']
                ce_ion_stat = self.statistics_dict['coordination_environments_ion_present']
                count_ions = self.statistics_dict['count_ion_present']
                for elmt, oxi_state, occ in site_species:
                    if elmt not in ion_stat:
                        ion_stat[elmt] = {}
                        count_ions[elmt] = {}
                    if oxi_state not in ion_stat[elmt]:
                        ion_stat[elmt][oxi_state] = {}
                        count_ions[elmt][oxi_state] = 0.0
                    count_ions[elmt][oxi_state] += occ
                    for ce_symbol, fraction in site_envs:
                        if fraction is None:
                            continue
                        if ce_symbol not in ion_stat[elmt][oxi_state]:
                            ion_stat[elmt][oxi_state][ce_symbol] = 0.0
                        ion_stat[elmt][oxi_state][ce_symbol] += occ * fraction
                        if ce_symbol not in ce_ion_stat:
                            ce_ion_stat[ce_symbol] = {}
                        if elmt not in ce_ion_stat[ce_symbol]:
                            ce_ion_stat[ce_symbol][elmt] = {}
                        if oxi_state not in ce_ion_stat[ce_symbol][elmt]:
                            ce_ion_stat[ce_symbol][elmt][oxi_state] = 0.0
                        ce_ion_stat[ce_symbol][elmt][oxi_state] += occ * fraction
    self.statistics_dict['anion_number'] = len(self.statistics_dict['anion_list'])
    self.statistics_dict['anion_atom_number'] = len(self.statistics_dict['anion_atom_list'])
    self.statistics_dict['cation_number'] = len(self.statistics_dict['cation_list'])
    self.statistics_dict['cation_atom_number'] = len(self.statistics_dict['cation_atom_list'])
    self.statistics_dict['neutral_number'] = len(self.statistics_dict['neutral_list'])
    self.statistics_dict['neutral_atom_number'] = len(self.statistics_dict['neutral_atom_list'])
    for elmt, envs in atom_stat.items():
        sumelement = count_atoms[elmt]
        fraction_atom_stat[elmt] = {env: fraction / sumelement for env, fraction in envs.items()}
    for ce_symbol, atoms in ce_atom_stat.items():
        sumsymbol = count_ce[ce_symbol]
        fraction_ce_atom_stat[ce_symbol] = {atom: fraction / sumsymbol for atom, fraction in atoms.items()}
    ion_stat = self.statistics_dict['ion_coordination_environments_present']
    fraction_ion_stat = self.statistics_dict['fraction_ion_coordination_environments_present']
    ce_ion_stat = self.statistics_dict['coordination_environments_ion_present']
    fraction_ce_ion_stat = self.statistics_dict['fraction_coordination_environments_ion_present']
    count_ions = self.statistics_dict['count_ion_present']
    for elmt, oxi_states_envs in ion_stat.items():
        fraction_ion_stat[elmt] = {}
        for oxi_state, envs in oxi_states_envs.items():
            sumspecie = count_ions[elmt][oxi_state]
            fraction_ion_stat[elmt][oxi_state] = {env: fraction / sumspecie for env, fraction in envs.items()}
    for ce_symbol, ions in ce_ion_stat.items():
        fraction_ce_ion_stat[ce_symbol] = {}
        sum_ce = np.sum([np.sum(list(oxi_states.values())) for elmt, oxi_states in ions.items()])
        for elmt, oxi_states in ions.items():
            fraction_ce_ion_stat[ce_symbol][elmt] = {oxistate: fraction / sum_ce for oxistate, fraction in oxi_states.items()}