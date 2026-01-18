import gzip
import struct
from collections import deque
from os.path import splitext
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.quaternions import Quaternions
def lammps_data_to_ase_atoms(data, colnames, cell, celldisp, pbc=False, atomsobj=Atoms, order=True, specorder=None, prismobj=None, units='metal'):
    """Extract positions and other per-atom parameters and create Atoms

    :param data: per atom data
    :param colnames: index for data
    :param cell: cell dimensions
    :param celldisp: origin shift
    :param pbc: periodic boundaries
    :param atomsobj: function to create ase-Atoms object
    :param order: sort atoms by id. Might be faster to turn off.
    Disregarded in case `id` column is not given in file.
    :param specorder: list of species to map lammps types to ase-species
    (usually .dump files to not contain type to species mapping)
    :param prismobj: Coordinate transformation between lammps and ase
    :type prismobj: Prism
    :param units: lammps units for unit transformation between lammps and ase
    :returns: Atoms object
    :rtype: Atoms

    """
    if 'id' in colnames:
        ids = data[:, colnames.index('id')].astype(int)
        if order:
            sort_order = np.argsort(ids)
            data = data[sort_order, :]
    if 'element' in colnames:
        elements = data[:, colnames.index('element')]
    elif 'type' in colnames:
        elements = data[:, colnames.index('type')].astype(int)
        if specorder:
            elements = [specorder[t - 1] for t in elements]
    else:
        raise ValueError('Cannot determine atom types form LAMMPS dump file')

    def get_quantity(labels, quantity=None):
        try:
            cols = [colnames.index(label) for label in labels]
            if quantity:
                return convert(data[:, cols].astype(float), quantity, units, 'ASE')
            return data[:, cols].astype(float)
        except ValueError:
            return None
    positions = None
    scaled_positions = None
    if 'x' in colnames:
        positions = get_quantity(['x', 'y', 'z'], 'distance')
    elif 'xs' in colnames:
        scaled_positions = get_quantity(['xs', 'ys', 'zs'])
    elif 'xu' in colnames:
        positions = get_quantity(['xu', 'yu', 'zu'], 'distance')
    elif 'xsu' in colnames:
        scaled_positions = get_quantity(['xsu', 'ysu', 'zsu'])
    else:
        raise ValueError('No atomic positions found in LAMMPS output')
    velocities = get_quantity(['vx', 'vy', 'vz'], 'velocity')
    charges = get_quantity(['q'], 'charge')
    forces = get_quantity(['fx', 'fy', 'fz'], 'force')
    quaternions = get_quantity(['c_q[1]', 'c_q[2]', 'c_q[3]', 'c_q[4]'])
    cell = convert(cell, 'distance', units, 'ASE')
    celldisp = convert(celldisp, 'distance', units, 'ASE')
    if prismobj:
        celldisp = prismobj.vector_to_ase(celldisp)
        cell = prismobj.update_cell(cell)
    if quaternions:
        out_atoms = Quaternions(symbols=elements, positions=positions, cell=cell, celldisp=celldisp, pbc=pbc, quaternions=quaternions)
    elif positions is not None:
        if prismobj:
            positions = prismobj.vector_to_ase(positions, wrap=True)
        out_atoms = atomsobj(symbols=elements, positions=positions, pbc=pbc, celldisp=celldisp, cell=cell)
    elif scaled_positions is not None:
        out_atoms = atomsobj(symbols=elements, scaled_positions=scaled_positions, pbc=pbc, celldisp=celldisp, cell=cell)
    if velocities is not None:
        if prismobj:
            velocities = prismobj.vector_to_ase(velocities)
        out_atoms.set_velocities(velocities)
    if charges is not None:
        out_atoms.set_initial_charges(charges)
    if forces is not None:
        if prismobj:
            forces = prismobj.vector_to_ase(forces)
        calculator = SinglePointCalculator(out_atoms, energy=0.0, forces=forces)
        out_atoms.calc = calculator
    for colname in colnames:
        if colname.startswith('f_') or colname.startswith('v_') or (colname.startswith('c_') and (not colname.startswith('c_q['))):
            out_atoms.new_array(colname, get_quantity([colname]), dtype='float')
    return out_atoms