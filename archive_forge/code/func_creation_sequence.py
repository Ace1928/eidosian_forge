from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def creation_sequence(degree_sequence, with_labels=False, compact=False):
    """
    Determines the creation sequence for the given threshold degree sequence.

    The creation sequence is a list of single characters 'd'
    or 'i': 'd' for dominating or 'i' for isolated vertices.
    Dominating vertices are connected to all vertices present when it
    is added.  The first node added is by convention 'd'.
    This list can be converted to a string if desired using "".join(cs)

    If with_labels==True:
    Returns a list of 2-tuples containing the vertex number
    and a character 'd' or 'i' which describes the type of vertex.

    If compact==True:
    Returns the creation sequence in a compact form that is the number
    of 'i's and 'd's alternating.
    Examples:
    [1,2,2,3] represents d,i,i,d,d,i,i,i
    [3,1,2] represents d,d,d,i,d,d

    Notice that the first number is the first vertex to be used for
    construction and so is always 'd'.

    with_labels and compact cannot both be True.

    Returns None if the sequence is not a threshold sequence
    """
    if with_labels and compact:
        raise ValueError('compact sequences cannot be labeled')
    if isinstance(degree_sequence, dict):
        ds = [[degree, label] for label, degree in degree_sequence.items()]
    else:
        ds = [[d, i] for i, d in enumerate(degree_sequence)]
    ds.sort()
    cs = []
    while ds:
        if ds[0][0] == 0:
            d, v = ds.pop(0)
            if len(ds) > 0:
                cs.insert(0, (v, 'i'))
            else:
                cs.insert(0, (v, 'd'))
            continue
        if ds[-1][0] != len(ds) - 1:
            return None
        d, v = ds.pop()
        cs.insert(0, (v, 'd'))
        ds = [[d[0] - 1, d[1]] for d in ds]
    if with_labels:
        return cs
    if compact:
        return make_compact(cs)
    return [v[1] for v in cs]