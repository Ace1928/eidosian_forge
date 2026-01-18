import os
import operator as op
import re
import warnings
from collections import OrderedDict
from os import path
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.dft.kpoints import kpoint_convert
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols, atomic_numbers
from ase.units import create_units
from ase.utils import iofunction
@iofunction('rU')
def read_espresso_in(fileobj):
    """Parse a Quantum ESPRESSO input files, '.in', '.pwi'.

    ESPRESSO inputs are generally a fortran-namelist format with custom
    blocks of data. The namelist is parsed as a dict and an atoms object
    is constructed from the included information.

    Parameters
    ----------
    fileobj : file | str
        A file-like object that supports line iteration with the contents
        of the input file, or a filename.

    Returns
    -------
    atoms : Atoms
        Structure defined in the input file.

    Raises
    ------
    KeyError
        Raised for missing keys that are required to process the file
    """
    data, card_lines = read_fortran_namelist(fileobj)
    if 'system' not in data:
        raise KeyError('Required section &SYSTEM not found.')
    elif 'ibrav' not in data['system']:
        raise KeyError('ibrav is required in &SYSTEM')
    elif data['system']['ibrav'] == 0:
        if 'celldm(1)' in data['system']:
            alat = data['system']['celldm(1)'] * units['Bohr']
        elif 'A' in data['system']:
            alat = data['system']['A']
        else:
            alat = None
        cell, cell_alat = get_cell_parameters(card_lines, alat=alat)
    else:
        alat, cell = ibrav_to_cell(data['system'])
    species_card = get_atomic_species(card_lines, n_species=data['system']['ntyp'])
    species_info = {}
    for ispec, (label, weight, pseudo) in enumerate(species_card):
        symbol = label_to_symbol(label)
        valence = get_valence_electrons(symbol, data, pseudo)
        magnet_key = 'starting_magnetization({0})'.format(ispec + 1)
        magmom = valence * data['system'].get(magnet_key, 0.0)
        species_info[symbol] = {'weight': weight, 'pseudo': pseudo, 'valence': valence, 'magmom': magmom}
    positions_card = get_atomic_positions(card_lines, n_atoms=data['system']['nat'], cell=cell, alat=alat)
    symbols = [label_to_symbol(position[0]) for position in positions_card]
    positions = [position[1] for position in positions_card]
    magmoms = [species_info[symbol]['magmom'] for symbol in symbols]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True, magmoms=magmoms)
    return atoms