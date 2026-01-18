import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
def distribute_cpus(size, comm):
    """Distribute cpus to tasks and calculators.

    Input:
    size: number of nodes per calculator
    comm: total communicator object

    Output:
    communicator for this rank, number of calculators, index for this rank
    """
    assert size <= comm.size
    assert comm.size % size == 0
    tasks_rank = comm.rank // size
    r0 = tasks_rank * size
    ranks = np.arange(r0, r0 + size)
    mycomm = comm.new_communicator(ranks)
    return (mycomm, comm.size // size, tasks_rank)