import networkx as nx
from networkx.utils import py_random_state
def choose_pref_attach(degs, seed):
    """Pick a random value, with a probability given by its weight.

    Returns a random choice among degs keys, each of which has a
    probability proportional to the corresponding dictionary value.

    Parameters
    ----------
    degs: dictionary
        It contains the possible values (keys) and the corresponding
        probabilities (values)
    seed: random state

    Returns
    -------
    v: object
        A key of degs or None if degs is empty
    """
    if len(degs) == 0:
        return None
    s = sum(degs.values())
    if s == 0:
        return seed.choice(list(degs.keys()))
    v = seed.random() * s
    nodes = list(degs.keys())
    i = 0
    acc = degs[nodes[i]]
    while v > acc:
        i += 1
        acc += degs[nodes[i]]
    return nodes[i]