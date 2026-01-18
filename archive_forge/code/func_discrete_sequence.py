import networkx as nx
from networkx.utils import py_random_state
@py_random_state(3)
def discrete_sequence(n, distribution=None, cdistribution=None, seed=None):
    """
    Return sample sequence of length n from a given discrete distribution
    or discrete cumulative distribution.

    One of the following must be specified.

    distribution = histogram of values, will be normalized

    cdistribution = normalized discrete cumulative distribution

    """
    import bisect
    if cdistribution is not None:
        cdf = cdistribution
    elif distribution is not None:
        cdf = cumulative_distribution(distribution)
    else:
        raise nx.NetworkXError('discrete_sequence: distribution or cdistribution missing')
    inputseq = [seed.random() for i in range(n)]
    seq = [bisect.bisect_left(cdf, s) - 1 for s in inputseq]
    return seq