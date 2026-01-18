import numpy as np
import PIL.Image
import pythreejs
import scipy.interpolate
import ipyvolume as ipv
from ipyvolume.datasets import UrlCached
def stars(N=1000, radius=100000, thickness=3, seed=42, color=[255, 240, 240]):
    rng = np.random.RandomState(seed)
    x, y, z = rng.normal(size=(3, N))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) / (radius + thickness * radius * np.random.random(N))
    x /= r
    y /= r
    z /= r
    s = ipv.scatter(x, y, z, texture=radial_sprite((64, 64), color), marker='square_2d', grow_limits=False, size=radius * 0.7 / 100)
    s.material.transparent = True
    return s