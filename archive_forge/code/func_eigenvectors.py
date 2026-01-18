from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def eigenvectors(creation_sequence):
    """
    Return a 2-tuple of Laplacian eigenvalues and eigenvectors
    for the threshold network with creation_sequence.
    The first value is a list of eigenvalues.
    The second value is a list of eigenvectors.
    The lists are in the same order so corresponding eigenvectors
    and eigenvalues are in the same position in the two lists.

    Notice that the order of the eigenvalues returned by eigenvalues(cs)
    may not correspond to the order of these eigenvectors.
    """
    ccs = make_compact(creation_sequence)
    N = sum(ccs)
    vec = [0] * N
    val = vec[:]
    dr = sum(ccs[::2])
    nn = ccs[0]
    vec[0] = [1.0 / sqrt(N)] * N
    val[0] = 0
    e = dr
    dr -= nn
    type_d = True
    i = 1
    dd = 1
    while dd < nn:
        scale = 1.0 / sqrt(dd * dd + i)
        vec[i] = i * [-scale] + [dd * scale] + [0] * (N - i - 1)
        val[i] = e
        i += 1
        dd += 1
    if len(ccs) == 1:
        return (val, vec)
    for nn in ccs[1:]:
        scale = 1.0 / sqrt(nn * i * (i + nn))
        vec[i] = i * [-nn * scale] + nn * [i * scale] + [0] * (N - i - nn)
        type_d = not type_d
        if type_d:
            e = i + dr
            dr -= nn
        else:
            e = dr
        val[i] = e
        st = i
        i += 1
        dd = 1
        while dd < nn:
            scale = 1.0 / sqrt(i - st + dd * dd)
            vec[i] = [0] * st + (i - st) * [-scale] + [dd * scale] + [0] * (N - i - 1)
            val[i] = e
            i += 1
            dd += 1
    return (val, vec)