import os
import warnings
import numpy as np
import ase
from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce
def read_netcdftrajectory(filename, index=-1):
    with NetCDFTrajectory(filename, mode='r') as traj:
        return traj[index]