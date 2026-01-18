import numpy as np
from scipy.constants import golden as phi
def octahedral(cls):
    g1 = tetrahedral(cls).as_quat()
    c = np.sqrt(2) / 2
    g2 = np.array([[+c, 0, 0, +c], [0, +c, 0, +c], [0, 0, +c, +c], [0, 0, -c, +c], [0, -c, 0, +c], [-c, 0, 0, +c], [0, +c, +c, 0], [0, -c, +c, 0], [+c, 0, +c, 0], [-c, 0, +c, 0], [+c, +c, 0, 0], [-c, +c, 0, 0]])
    return cls.from_quat(np.concatenate((g1, g2)))