from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.serialization import dumpfn, loadfn
from tqdm import tqdm
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine, Spin
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Orbital
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
def compute_properties_doping(self, doping, temp_r=None) -> None:
    """Calculate all the properties w.r.t. the doping levels in input.

        Args:
            doping: numpy array specifying the doping levels
            temp_r: numpy array specifying the temperatures

        When executed, it add the following variable at the BztTransportProperties
        object:
            Conductivity_doping, Seebeck_doping, Kappa_doping, Power_Factor_doping,
            cond_Effective_mass_doping are dictionaries with 'n' and 'p' keys and
            arrays of dim (len(temp_r),len(doping),3,3) as values.
            Carriers_conc_doping: carriers concentration for each doping level and T.
            mu_doping_eV: the chemical potential correspondent to each doping level.
        """
    if temp_r is None:
        temp_r = self.temp_r
    self.Conductivity_doping, self.Seebeck_doping, self.Kappa_doping, self.Carriers_conc_doping = ({}, {}, {}, {})
    self.Power_Factor_doping, self.Effective_mass_doping = ({}, {})
    mu_doping = {}
    doping_carriers = [dop * (self.volume / (units.Meter / 100.0) ** 3) for dop in doping]
    for dop_type in ['n', 'p']:
        sbk = np.zeros((len(temp_r), len(doping), 3, 3))
        cond = np.zeros((len(temp_r), len(doping), 3, 3))
        kappa = np.zeros((len(temp_r), len(doping), 3, 3))
        hall = np.zeros((len(temp_r), len(doping), 3, 3, 3))
        dc = np.zeros((len(temp_r), len(doping)))
        if dop_type == 'p':
            doping_carriers = [-dop for dop in doping_carriers]
        mu_doping[dop_type] = np.zeros((len(temp_r), len(doping)))
        for idx_t, temp in enumerate(temp_r):
            for idx_d, dop_car in enumerate(doping_carriers):
                mu_doping[dop_type][idx_t, idx_d] = BL.solve_for_mu(self.epsilon, self.dos, self.nelect + dop_car, temp, self.dosweight, True, False)
            N, L0, L1, L2, Lm11 = BL.fermiintegrals(self.epsilon, self.dos, self.vvdos, mur=mu_doping[dop_type][idx_t], Tr=np.array([temp]), dosweight=self.dosweight)
            cond[idx_t], sbk[idx_t], kappa[idx_t], hall[idx_t] = BL.calc_Onsager_coefficients(L0, L1, L2, mu_doping[dop_type][idx_t], np.array([temp]), self.volume, Lm11)
            dc[idx_t] = self.nelect + N
        self.Conductivity_doping[dop_type] = cond * self.CRTA
        self.Seebeck_doping[dop_type] = sbk * 1000000.0
        self.Kappa_doping[dop_type] = kappa * self.CRTA
        self.Carriers_conc_doping[dop_type] = dc / (self.volume / (units.Meter / 100.0) ** 3)
        self.Power_Factor_doping[dop_type] = sbk @ sbk @ cond * self.CRTA * 1000.0
        cond_eff_mass = np.zeros((len(temp_r), len(doping), 3, 3))
        for idx_t in range(len(temp_r)):
            for idx_d, dop in enumerate(doping):
                try:
                    cond_eff_mass[idx_t, idx_d] = np.linalg.inv(cond[idx_t, idx_d]) * dop * units.qe_SI ** 2 / units.me_SI * 1000000.0
                except np.linalg.LinAlgError:
                    pass
        self.Effective_mass_doping[dop_type] = cond_eff_mass
    self.doping = doping
    self.mu_doping = mu_doping
    self.mu_doping_eV = {k: v / units.eV - self.efermi for k, v in mu_doping.items()}
    self.contain_props_doping = True