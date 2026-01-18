from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def degree_sequence(creation_sequence):
    """
    Return degree sequence for the threshold graph with the given
    creation sequence
    """
    cs = creation_sequence
    seq = []
    rd = cs.count('d')
    for i, sym in enumerate(cs):
        if sym == 'd':
            rd -= 1
            seq.append(rd + i)
        else:
            seq.append(rd)
    return seq