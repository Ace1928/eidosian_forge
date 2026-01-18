import numpy as np
def mindist(pos1, pos2):
    n1 = len(pos1)
    n2 = len(pos2)
    idx1 = np.arange(n1).repeat(n2)
    idx2 = np.tile(np.arange(n2), n1)
    return np.sqrt(((pos1[idx1] - pos2[idx2]) ** 2).sum(axis=1).min())