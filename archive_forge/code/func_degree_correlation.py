from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def degree_correlation(creation_sequence):
    """
    Return the degree-degree correlation over all edges.
    """
    cs = creation_sequence
    s1 = 0
    s2 = 0
    s3 = 0
    m = 0
    rd = cs.count('d')
    rdi = [i for i, sym in enumerate(cs) if sym == 'd']
    ds = degree_sequence(cs)
    for i, sym in enumerate(cs):
        if sym == 'd':
            if i != rdi[0]:
                print('Logic error in degree_correlation', i, rdi)
                raise ValueError
            rdi.pop(0)
        degi = ds[i]
        for dj in rdi:
            degj = ds[dj]
            s1 += degj * degi
            s2 += degi ** 2 + degj ** 2
            s3 += degi + degj
            m += 1
    denom = 2 * m * s2 - s3 * s3
    numer = 4 * m * s1 - s3 * s3
    if denom == 0:
        if numer == 0:
            return 1
        raise ValueError(f'Zero Denominator but Numerator is {numer}')
    return numer / denom