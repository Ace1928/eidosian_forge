import os
import time
import math
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from ase.io import write, read
from ase.geometry.cell import cell_to_cellpar
from ase.data import covalent_radii
from ase.ga import get_neighbor_list
def get_nnmat(atoms, mic=False):
    """Calculate the nearest neighbor matrix as specified in
    S. Lysgaard et al., Top. Catal., 2014, 57 (1-4), pp 33-39

    Returns an array of average numbers of nearest neighbors
    the order is determined by self.elements.
    Example: self.elements = ["Cu", "Ni"]
    get_nnmat returns a single list [Cu-Cu bonds/N(Cu),
    Cu-Ni bonds/N(Cu), Ni-Cu bonds/N(Ni), Ni-Ni bonds/N(Ni)]
    where N(element) is the number of atoms of the type element
    in the atoms object.

    The distance matrix can be quite costly to calculate every
    time nnmat is required (and disk intensive if saved), thus
    it makes sense to calculate nnmat along with e.g. the
    potential energy and save it in atoms.info['data']['nnmat'].
    """
    if 'data' in atoms.info and 'nnmat' in atoms.info['data']:
        return atoms.info['data']['nnmat']
    elements = sorted(set(atoms.get_chemical_symbols()))
    nnmat = np.zeros((len(elements), len(elements)))
    dm = atoms.get_all_distances(mic=mic)
    nndist = get_nndist(atoms, dm) + 0.2
    for i in range(len(atoms)):
        row = [j for j in range(len(elements)) if atoms[i].symbol == elements[j]][0]
        neighbors = [j for j in range(len(dm[i])) if dm[i][j] < nndist]
        for n in neighbors:
            column = [j for j in range(len(elements)) if atoms[n].symbol == elements[j]][0]
            nnmat[row][column] += 1
    for i, el in enumerate(elements):
        nnmat[i] /= len([j for j in range(len(atoms)) if atoms[int(j)].symbol == el])
    nnlist = np.reshape(nnmat, len(nnmat) ** 2)
    return nnlist