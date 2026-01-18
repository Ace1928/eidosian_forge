import numpy as np
def plot2d(self, ix=0, iy=1, clf=True):
    """
        Generates a 2-dimensional plot of the data set and principle components
        using matplotlib.

        ix specifies which p-dimension to put on the x-axis of the plot
        and iy specifies which to put on the y-axis (0-indexed)
        """
    import matplotlib.pyplot as plt
    x, y = (self.N[:, ix], self.N[:, iy])
    if clf:
        plt.clf()
    plt.scatter(x, y)
    vals, evs = self.getEigensystem()
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    dx, dy = (xu - xl, yu - yl)
    for val, vec, c in zip(vals, evs.T, self._colors):
        plt.arrow(0, 0, val * vec[ix], val * vec[iy], head_width=0.05 * (dx * dy / 4) ** 0.5, fc=c, ec=c)
    if self.names is not None:
        plt.xlabel('$' + self.names[ix] + '/\\sigma$')
        plt.ylabel('$' + self.names[iy] + '/\\sigma$')