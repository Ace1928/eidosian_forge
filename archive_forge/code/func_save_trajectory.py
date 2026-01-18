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
def save_trajectory(confid, trajectory, folder):
    """Saves traj files to the database folder.
    This method should never be used directly,
    but only through the DataConnection object.
    """
    fname = os.path.join(folder, 'traj%05d.traj' % confid)
    write(fname, trajectory)
    return fname