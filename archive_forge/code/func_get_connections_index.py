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
def get_connections_index(atoms, max_conn=5, no_count_types=None):
    """This method returns a dictionary where each key value are a
    specific number of neighbors and list of atoms indices with
    that amount of neighbors respectively. The method utilizes the
    neighbor list and hence inherit the restrictions for
    neighbors. Option added to remove connections between
    defined atom types.

    Parameters
    ----------

    atoms : Atoms object
        The connections will be counted using this supplied Atoms object

    max_conn : int
        Any atom with more connections than this will be counted as
        having max_conn connections.
        Default 5

    no_count_types : list or None
        List of atomic numbers that should be excluded in the count.
        Default None (meaning all atoms count).
    """
    conn = get_neighbor_list(atoms)
    if conn is None:
        conn = get_neighborlist(atoms)
    if no_count_types is None:
        no_count_types = []
    conn_index = {}
    for i in range(len(atoms)):
        if atoms[i].number not in no_count_types:
            cconn = min(len(conn[i]), max_conn - 1)
            if cconn not in conn_index:
                conn_index[cconn] = []
            conn_index[cconn].append(i)
    return conn_index