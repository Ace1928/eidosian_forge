from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
@py_random_state(2)
def random_threshold_sequence(n, p, seed=None):
    """
    Create a random threshold sequence of size n.
    A creation sequence is built by randomly choosing d's with
    probability p and i's with probability 1-p.

    s=nx.random_threshold_sequence(10,0.5)

    returns a threshold sequence of length 10 with equal
    probably of an i or a d at each position.

    A "random" threshold graph can be built with

    G=nx.threshold_graph(s)

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    if not 0 <= p <= 1:
        raise ValueError('p must be in [0,1]')
    cs = ['d']
    for i in range(1, n):
        if seed.random() < p:
            cs.append('d')
        else:
            cs.append('i')
    return cs