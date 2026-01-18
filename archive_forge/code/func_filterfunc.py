import pygsp
import numpy as np
def filterfunc(x):
    return np.exp(-beta * np.abs(x / graph.lmax - offset) ** order)