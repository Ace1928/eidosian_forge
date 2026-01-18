import warnings
import numpy as np
from numpy import cos, sin, pi
def example_ylm(m=0, n=2, shape=128, limits=[-4, 4], draw=True, show=True, **kwargs):
    """Show a spherical harmonic."""
    import ipyvolume.pylab as p3
    __, __, __, r, theta, phi = xyz(shape=shape, limits=limits, spherical=True)
    radial = np.exp(-(r - 2) ** 2)
    data = np.abs(scipy.special.sph_harm(m, n, theta, phi) ** 2) * radial
    if draw:
        vol = p3.volshow(data=data, **kwargs)
        if show:
            p3.show()
        return vol
    else:
        return data