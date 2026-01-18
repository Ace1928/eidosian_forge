import numpy as np
from scipy.constants import golden as phi
def icosahedral(cls):
    g1 = tetrahedral(cls).as_quat()
    a = 0.5
    b = 0.5 / phi
    c = phi / 2
    g2 = np.array([[+a, +b, +c, 0], [+a, +b, -c, 0], [+a, +c, 0, +b], [+a, +c, 0, -b], [+a, -b, +c, 0], [+a, -b, -c, 0], [+a, -c, 0, +b], [+a, -c, 0, -b], [+a, 0, +b, +c], [+a, 0, +b, -c], [+a, 0, -b, +c], [+a, 0, -b, -c], [+b, +a, 0, +c], [+b, +a, 0, -c], [+b, +c, +a, 0], [+b, +c, -a, 0], [+b, -a, 0, +c], [+b, -a, 0, -c], [+b, -c, +a, 0], [+b, -c, -a, 0], [+b, 0, +c, +a], [+b, 0, +c, -a], [+b, 0, -c, +a], [+b, 0, -c, -a], [+c, +a, +b, 0], [+c, +a, -b, 0], [+c, +b, 0, +a], [+c, +b, 0, -a], [+c, -a, +b, 0], [+c, -a, -b, 0], [+c, -b, 0, +a], [+c, -b, 0, -a], [+c, 0, +a, +b], [+c, 0, +a, -b], [+c, 0, -a, +b], [+c, 0, -a, -b], [0, +a, +c, +b], [0, +a, +c, -b], [0, +a, -c, +b], [0, +a, -c, -b], [0, +b, +a, +c], [0, +b, +a, -c], [0, +b, -a, +c], [0, +b, -a, -c], [0, +c, +b, +a], [0, +c, +b, -a], [0, +c, -b, +a], [0, +c, -b, -a]])
    return cls.from_quat(np.concatenate((g1, g2)))