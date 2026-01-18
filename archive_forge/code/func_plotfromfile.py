import numpy as np
from collections import namedtuple
from ase.geometry import find_mic
def plotfromfile(*fnames):
    from ase.io import read
    nplots = len(fnames)
    for i, fname in enumerate(fnames):
        images = read(fname, ':')
        import matplotlib.pyplot as plt
        plt.subplot(nplots, 1, 1 + i)
        force_curve(images)
    plt.show()