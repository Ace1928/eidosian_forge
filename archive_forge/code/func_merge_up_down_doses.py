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
def merge_up_down_doses(dos_up, dos_dn):
    """Merge the up and down DOSs.

    Args:
        dos_up: Up DOS.
        dos_dn: Down DOS

    Returns:
        CompleteDos object
    """
    warnings.warn('This function is not useful anymore. VasprunBSLoader deals with spin case.')
    cdos = Dos(dos_up.efermi, dos_up.energies, {Spin.up: dos_up.densities[Spin.up], Spin.down: dos_dn.densities[Spin.down]})
    if hasattr(dos_up, 'pdos') and hasattr(dos_dn, 'pdos'):
        pdoss = {}
        for site in dos_up.pdos:
            pdoss.setdefault(site, {})
            for orb in dos_up.pdos[site]:
                pdoss[site].setdefault(orb, {})
                pdoss[site][orb][Spin.up] = dos_up.pdos[site][orb][Spin.up]
                pdoss[site][orb][Spin.down] = dos_dn.pdos[site][orb][Spin.down]
        cdos = CompleteDos(dos_up.structure, total_dos=cdos, pdoss=pdoss)
    return cdos